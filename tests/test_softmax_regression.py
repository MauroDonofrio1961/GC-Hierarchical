import numpy as np
from scipy.special import softmax


def test_three_component_softmax_is_normalized():
    for logits in (
        [0.0, 0.0, 0.0],
        [4.0, -4.0, 0.0],
        [-4.0, 4.0, 0.0],
        [2.3, 1.7, 0.0],
    ):
        probabilities = softmax(np.asarray(logits, dtype=float))
        assert np.all(probabilities > 0.0)
        assert np.isclose(probabilities.sum(), 1.0, atol=1e-12)
