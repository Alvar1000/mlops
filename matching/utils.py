import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.sqrt(np.sum(np.square(vector)))
    if norm > 0.001:
        return vector / norm
    else:
        return vector
