from sklearn.model_selection import RepeatedStratifiedKFold


def split_data(y, df, n_splits=5, n_repeats=5, random_state=42):
    """
    Generates repeated stratified K-fold splits for stable cross-validation.

    Reduces variance in performance estimates, which is critical for
    high-dimensional data (D >> N).

    Args:
        y (array-like): Target labels (e.g., 1 for hallucination, 0 for fact).
        df (DataFrame/ndarray): Feature dataset.
        n_splits (int, optional): Number of folds (default: 5).
        n_repeats (int, optional): Number of repetitions (default: 5).
        random_state (int, optional): Random seed (default: 42).

    Returns:
        list[tuple]: A list of `(train_idx, None, test_idx)` tuples.
        The validation index is `None` because the downstream Bootstrap
        ensemble does not require independent threshold tuning.
    """
    y_array = y.values if hasattr(y, "values") else y

    # Repeated stratified K-fold: variance reduction through repetitions
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    splits = []
    for train_idx, test_idx in rskf.split(df, y_array):
        # The validation fold is set to None (no threshold tuning required)
        splits.append((train_idx, None, test_idx))

    return splits
