import numpy as np
import pytest

from modules.significance import compare_forecasters, diebold_mariano_test, friedman_nemenyi_test


def test_dm_test_identical_losses_gives_no_significant_difference():
    losses = np.array([1.0, 2.0, 1.5, 2.5, 1.8])
    result = diebold_mariano_test(losses, losses.copy())
    assert result["p_value"] == pytest.approx(1.0)
    assert result["mean_diff"] == pytest.approx(0.0)


def test_dm_test_detects_a_consistently_worse_forecaster():
    rng = np.random.default_rng(0)
    loss_a = rng.normal(1.0, 0.1, size=30)
    loss_b = rng.normal(3.0, 0.1, size=30)  # consistently much worse
    result = diebold_mariano_test(loss_a, loss_b)
    assert result["p_value"] < 0.01
    assert result["mean_diff"] < 0


def test_dm_test_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        diebold_mariano_test([1.0], [2.0])


def test_friedman_nemenyi_requires_at_least_three_methods():
    import pandas as pd
    with pytest.raises(ValueError):
        friedman_nemenyi_test(pd.DataFrame({"a": [1, 2], "b": [2, 1]}))


def test_friedman_nemenyi_ranks_consistently_worse_method_last():
    import pandas as pd
    loss_matrix = pd.DataFrame({
        "best": [1.0, 1.1, 0.9, 1.2, 1.0],
        "middle": [2.0, 2.1, 1.9, 2.2, 2.0],
        "worst": [3.0, 3.1, 2.9, 3.2, 3.0],
    })
    result = friedman_nemenyi_test(loss_matrix)
    assert result["avg_ranks"]["best"] < result["avg_ranks"]["middle"] < result["avg_ranks"]["worst"]
    assert result["p_value"] < 0.05
    assert result["critical_difference"] > 0


def test_compare_forecasters_dispatches_by_forecaster_count():
    two_way = compare_forecasters({"a": [1.0, 2.0, 3.0], "b": [1.1, 2.1, 3.1]})
    assert two_way["test"] == "diebold_mariano"

    three_way = compare_forecasters({
        "a": [1.0, 1.1, 0.9], "b": [2.0, 2.1, 1.9], "c": [3.0, 3.1, 2.9],
    })
    assert three_way["test"] == "friedman_nemenyi"


def test_compare_forecasters_rejects_mismatched_lengths_for_three_way():
    with pytest.raises(ValueError):
        compare_forecasters({"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0], "c": [1.0, 2.0, 3.0]})


def test_compare_forecasters_requires_at_least_two():
    with pytest.raises(ValueError):
        compare_forecasters({"a": [1.0, 2.0]})
