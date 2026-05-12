import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class HallucinationProbe:
    """
    A classifier for LLM hallucination detection based on hidden states (Representation Engineering).

    Implements a Multi-view Feature Fusion strategy: the global feature space (D=3603)
    is partitioned into three independent representations (Views), each training
    its own ensemble of logistic regressions. The final prediction is formed
    by averaging the output probabilities (Soft Voting / Late Fusion).

    To combat the curse of dimensionality on a small dataset, extreme
    L2-regularization (C=0.003) and bagging (Bootstrap Aggregating) are applied.
    """

    def __init__(self):
        # Feature slices are strictly synchronized with aggregation_and_feature_extraction output
        # View A: Max spikes (L12, L13) + Metadata + Global Geometry (1801 features)
        self.slice_a = slice(0, 1801)
        # View B: Integral semantic drift L14 + Metadata (901 features)
        self.slice_b = slice(1801, 1801 + 901)
        # View C: Integral semantic drift L15 + Metadata (901 features)
        self.slice_c = slice(1801 + 901, 1801 + 901 + 901)

        self.threshold = 0.5

        def create_view_ensemble():
            """
            Creates a base pipeline and wraps it in a Bootstrap ensemble.

            Returns:
                BaggingClassifier: An ensemble of 30 logistic regressions.
            """
            base_pipe = Pipeline(
                [
                    # Standardize features to zero mean and unit variance
                    ("scaler", StandardScaler()),
                    # Extremely strong L2-regularization to suppress noise in D >> N
                    (
                        "lr",
                        LogisticRegression(
                            C=0.003,
                            penalty="l2",
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            )
            # Bagging reduces model variance by training on bootstrap samples
            return BaggingClassifier(
                estimator=base_pipe,
                n_estimators=30,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            )

        # Independent ensembles for each semantic feature block
        self.model_A = create_view_ensemble()
        self.model_B = create_view_ensemble()
        self.model_C = create_view_ensemble()

    def fit(self, X, y):
        """
        Trains independent ensembles on corresponding feature subspaces.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, 3603).
            y (np.ndarray): Target labels vector (1 - hallucination, 0 - fact).
        """
        # Sanitize from potential computational artifacts (NaN/Inf)
        X_clean = np.nan_to_num(X).astype(np.float32)
        # Separate training ensures noisy Views do not drown out signals in others
        self.model_A.fit(X_clean[:, self.slice_a], y)
        self.model_B.fit(X_clean[:, self.slice_b], y)
        self.model_C.fit(X_clean[:, self.slice_c], y)
        return self

    def predict_proba(self, X):
        """
        Calculates class probabilities by averaging ensemble predictions.
        """
        X_clean = np.nan_to_num(X).astype(np.float32)
        prob_a = self.model_A.predict_proba(X_clean[:, self.slice_a])
        prob_b = self.model_B.predict_proba(X_clean[:, self.slice_b])
        prob_c = self.model_C.predict_proba(X_clean[:, self.slice_c])
        # Late Fusion: equal weight probability averaging (Soft Voting)
        return (prob_a + prob_b + prob_c) / 3.0

    def predict(self, X):
        """
        Predicts binary class labels based on the defined threshold.
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= self.threshold).astype(int)
