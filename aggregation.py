import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path

# Global state for persistent data across function calls
_tokenizer = None
_prompt_lengths = None
_call_counter = 0


def _initialize_tokenizer():
    """
    Initializes the global Qwen2.5 tokenizer and pre-computes prompt lengths.

    Acts as a singleton to load the tokenizer and cache the token counts for all
    prompts in the local train/test datasets into `_prompt_lengths`. This offline
    pre-processing enables O(1) "Smart Masking" (exact boundary detection between
    prompt and response) without the overhead of runtime re-tokenization.

    Globals Modified:
        _tokenizer: The initialized Hugging Face tokenizer instance.
        _prompt_lengths (list[int]): Cached exact lengths for every dataset prompt.
    """

    global _tokenizer, _prompt_lengths
    if _tokenizer is not None:
        return

    # Locate data files relative to the current script
    repo_root = Path(__file__).resolve().parent
    train_path = repo_root / "data" / "dataset.csv"
    test_path = repo_root / "data" / "test.csv"

    # Load Qwen2.5 tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )

    # Read datasets to extract prompt texts
    train_df = (
        pd.read_csv(train_path) if train_path.exists() else pd.DataFrame({"prompt": []})
    )
    test_df = (
        pd.read_csv(test_path) if test_path.exists() else pd.DataFrame({"prompt": []})
    )
    all_prompts = list(train_df.get("prompt", [])) + list(test_df.get("prompt", []))

    # Store token counts for every prompt to perform exact slicing in aggregation
    _prompt_lengths = []
    for prompt in all_prompts:
        tokens = _tokenizer.encode(prompt, add_special_tokens=False)
        _prompt_lengths.append(len(tokens))


def aggregation_and_feature_extraction(hidden_states, mask, use_geometric=True):
    """
    Aggregates the hidden states of an LLM (Qwen2.5-0.5B) and extracts a feature vector
    of dimension 3603 for hallucination detection using Representation Engineering.

    The pipeline implements a Multi-view Feature Fusion strategy and "Smart Masking"
    (stripping the prompt and the final EOS token to suppress structural noise).
    The final feature space (D=3603) is formed by concatenating three views:

    1. View A (1801 features):
       - Local factual singularities: Max-pooling across response tokens
         on layers 12 and 13 (896 + 896 = 1792 features).
       - Length metadata: Absolute and logarithmic context sizes (5 features).
       - Global macro-geometry: Evaluation of semantic stability across reference
         layers [0, 6, 12, 18, 23] using 4 metrics (norm_cv, norm_ratio, inter_cos_min,
         inter_cos_mean).

    2. View B (901 features):
       - Integral semantic drift: Mean-pooling across response tokens on
         layer 14 (896 features).
       - Duplicated length metadata (5 features).

    3. View C (901 features):
       - Integral semantic drift: Mean-pooling across response tokens on
         layer 15 (896 features).
       - Duplicated length metadata (5 features).

    Args:
        hidden_states (torch.Tensor | list | tuple): Tensor of the model's hidden states.
            Expected shape: (num_layers, seq_len, hidden_dim) or
            (num_layers, 1, seq_len, hidden_dim).
        mask (torch.Tensor): Attention mask for active tokens,
            where values > 0 indicate real tokens in the sequence.
        use_geometric (bool, optional): Flag to use geometric features

    Returns:
        torch.Tensor: A 1D feature tensor of shape (3603),
        sanitized of NaNs (replaced with 0.0) and moved to the CPU.
    """
    global _call_counter
    _initialize_tokenizer()

    # Standardize the hidden_states tensor to [Layers, Seq_Len, Hidden_Dim]
    if isinstance(hidden_states, (list, tuple)):
        hidden_states = torch.stack(hidden_states)
    if hidden_states.dim() == 4:
        hidden_states = hidden_states.squeeze(1)

    # Filter active (non-padded) tokens
    active_idx = torch.where(mask > 0)[0]
    n_real = len(active_idx)

    # Retrieve the pre-computed prompt length for the current sample
    prompt_len = _prompt_lengths[_call_counter]
    _call_counter += 1

    # Isolate the response span. We exclude the final token (EOS) as it contains
    # structural noise rather than factual content.
    resp_start = min(prompt_len, n_real - 1)
    resp_end = max(resp_start, n_real - 1)

    # Fallback logic for truncated sequences
    if resp_start >= resp_end:
        resp_start = max(0, n_real - 2)
        resp_end = n_real - 1
    resp_idx = active_idx[resp_start:resp_end]

    # Ensure indices are never empty
    if len(resp_idx) == 0:
        resp_idx = active_idx[-1:]

    # Mid-layer Max-Pooling (Layers 12 & 13)
    # Max-pooling captures factual 'spikes' in specific neurons that indicate uncertainty.
    h12 = hidden_states[12, resp_idx, :].float()
    h13 = hidden_states[13, resp_idx, :].float()
    view_a_l12 = h12.max(dim=0)[0]
    view_a_l13 = h13.max(dim=0)[0]

    # Statistical length features to account for the 'Alignment Tax' (long responses)
    resp_n = len(resp_idx)
    view_a_lengths = torch.tensor(
        [n_real, resp_n, prompt_len, np.log1p(resp_n), np.log1p(prompt_len)],
        device=h12.device,
    )

    # Geometry: Track energy and trajectory stability
    # Coefficient variation of norms on the final layer
    last_layer = hidden_states[23, active_idx, :].float()
    token_norms = torch.norm(last_layer, p=2, dim=-1)
    norm_cv = token_norms.std() / (token_norms.mean() + 1e-8)

    # Ratio of energy growth from input to output
    first_layer = hidden_states[0, active_idx, :].float()
    norm_ratio = torch.norm(last_layer.mean(0)) / (
        torch.norm(first_layer.mean(0)) + 1e-8
    )

    # Compute inter-layer cosine similarities across the network depth
    layer_means = []
    for layer_idx in [0, 6, 12, 18, 23]:
        layer_means.append(hidden_states[layer_idx, active_idx, :].float().mean(dim=0))
    inter_cos = []
    for i in range(len(layer_means) - 1):
        inter_cos.append(
            torch.nn.functional.cosine_similarity(
                layer_means[i], layer_means[i + 1], dim=0
            )
        )

    view_a_geom = torch.tensor(
        [
            norm_cv.item(),
            norm_ratio.item(),
            min([c.item() for c in inter_cos]),
            sum([c.item() for c in inter_cos]) / len(inter_cos),
        ],
        device=h12.device,
    )

    # 1801 dimensions
    view_a = torch.cat([view_a_l12, view_a_l13, view_a_lengths, view_a_geom])

    # Layer 14 Mean-Pooling (901 dimensions)
    h14 = hidden_states[14, resp_idx, :].float()
    view_b = torch.cat([h14.mean(dim=0), view_a_lengths])

    # Layer 15 Mean-Pooling (901 dimensions)
    h15 = hidden_states[15, resp_idx, :].float()
    view_c = torch.cat([h15.mean(dim=0), view_a_lengths])

    # Final fusion into a single 3603-dimensional vector
    features = torch.cat([view_a, view_b, view_c])

    # Clean up any potential mathematical artifacts
    return torch.nan_to_num(features, nan=0.0).cpu()
