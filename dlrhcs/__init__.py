"""dlrhcs -- Cross-Fitted Debiased Inference for Dynamic Panels with Low-Rank
Heterogeneous Coefficients (replication package)."""
from . import design, dgp, factorridge, folds, onestep, pipeline, ranks, targets

__all__ = ["design", "dgp", "factorridge", "folds", "onestep", "pipeline",
           "ranks", "targets"]
__version__ = "0.1.0"
