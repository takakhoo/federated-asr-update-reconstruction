import numpy as np
import pytest
from scripts.least_squares import recover_features


def test_full_rank_exact_and_shapes():
    rng = np.random.default_rng(7)
    g, truth = rng.normal(size=(4, 7)), rng.normal(size=(4, 3))
    recovered, _, report = recover_features(g.T@truth, g.reshape(2, 2, 7))
    np.testing.assert_allclose(recovered.reshape(4, 3), truth, atol=1e-12)
    assert recovered.shape == (2, 2, 3) and report["nullity"] == 0
    assert report["mean_identifiable"]


def test_underdetermined_is_projection_not_truth():
    rng = np.random.default_rng(7)
    g, truth = rng.normal(size=(20, 5)), rng.normal(size=(20, 3))
    recovered, _, report = recover_features(g.T@truth, g)
    np.testing.assert_allclose(g.T@recovered, g.T@truth, atol=1e-12)
    assert np.linalg.norm(recovered-truth) > 1
    assert report["nullity"] == 15 and not report["mean_identifiable"]


def test_zero_rows_preserve_frame_positions():
    g = np.array([[1., 0], [0, 0], [0, 1]])
    recovered, _, report = recover_features(np.eye(2), g)
    assert recovered.shape == (3, 2)
    assert np.all(recovered[1] == 0)
    assert report["zero_gradient_rows"] == [1]
    all_zero, _, report = recover_features(np.zeros((2, 3)), np.zeros((4, 2)))
    assert np.all(all_zero == 0) and report["rank"] == 0


def test_ridge_solves_regularized_normal_equations():
    rng = np.random.default_rng(7)
    g, dw = rng.normal(size=(8, 4)), rng.normal(size=(4, 3))
    recovered, _, _ = recover_features(dw, g, ridge=.1)
    np.testing.assert_allclose((g@g.T + .1*np.eye(8))@recovered, g@dw, atol=1e-12)


def test_invalid_inputs():
    for g in (np.ones((3, 4)), np.full((3, 2), np.nan)):
        with pytest.raises(ValueError):
            recover_features(np.ones((2, 3)), g)
    with pytest.raises(ValueError):
        recover_features(np.ones((2, 3)), np.ones((3, 2)), ridge=-1)


def test_cli_preserves_both_plots_and_arrays(tmp_path):
    import torch
    from scripts.reconstruct_from_grads import _load_core_dependencies, run_ls_mode
    _load_core_dependencies()
    blob = tmp_path/"fixture.pt"
    torch.save({"dW": torch.eye(2), "dlogits": torch.eye(2)[None]}, blob)
    run_ls_mode(str(blob), str(tmp_path/"out"))
    assert all((tmp_path/"out"/name).exists() for name in ("features.png", "constant-fit.png", "reconstruction.npz", "diagnostics.json"))
