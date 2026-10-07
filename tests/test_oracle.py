import numpy as np
import pytest

from src.adts.oracle import dynamic_oracle_actions
from src.adts.envs import abrupt_means
from project_publication.pipeline import build_publication_environment_suite


def test_dynamic_oracle_tracks_switches_and_breaks_ties_by_index():
    # Time-major input: the best arm switches; a tie picks the first arm.
    actions, means = dynamic_oracle_actions(np.array([[0.1, 0.8], [0.9, 0.2], [0.5, 0.5]]))
    np.testing.assert_array_equal(actions, [1, 0, 0])
    np.testing.assert_array_equal(means, [0.8, 0.9, 0.5])


def test_abrupt_environment_jump_and_cycle_boundaries():
    means = abrupt_means(horizon=251)
    np.testing.assert_array_equal(means[49], [0, 0, 0, 0])
    np.testing.assert_array_equal(means[50], [0.1, 0, 0, 0])
    np.testing.assert_array_equal(means[200], [0.1, 0.37, 0.63, 0.9])
    np.testing.assert_array_equal(means[250], [0, 0, 0, 0])
    with pytest.raises(ValueError, match="exactly four"):
        build_publication_environment_suite(120, 5)
