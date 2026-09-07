import numpy as np
import pytest
import torch

from core import aerca_verifier as verifier
from core.models.senn import SENNGC
from utils.reproducibility import seed_everything, split_verifier_data


def test_single_chunk_split_is_disjoint_and_copied():
    data = np.arange(200, dtype=np.float32).reshape(1, 100, 2)
    fit, score = split_verifier_data(data, 2)
    assert fit.shape == (1, 80, 2)
    assert score.shape == (1, 20, 2)
    assert fit.max() < score.min()
    fit[:] = -1
    assert data.min() == 0


def test_multiple_chunk_split_preserves_order():
    data = np.arange(5 * 20 * 2).reshape(5, 20, 2)
    fit, score = split_verifier_data(data, 2)
    np.testing.assert_array_equal(fit, data[:4])
    np.testing.assert_array_equal(score, data[4:])


@pytest.mark.parametrize(
    "data,n",
    [
        ([], 2),
        (np.zeros((10, 2)), 2),
        (np.zeros((1, 19, 2)), 2),
        (np.zeros((1, 20, 1)), 1),
        (np.full((1, 20, 2), np.nan), 2),
        (np.zeros((1, 20, 3)), 2),
    ],
)
def test_bad_data_fails_early(data, n):
    with pytest.raises(ValueError):
        split_verifier_data(data, n)


def test_seed_repeatability():
    seed_everything(12)
    a, b = np.random.random(3), torch.rand(3)
    seed_everything(12)
    np.testing.assert_array_equal(a, np.random.random(3))
    assert torch.equal(b, torch.rand(3))
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_no_zero_or_diagonal_baseline_edges():
    names = ["A", "B"]
    assert verifier.build_baseline_graph(np.zeros((2, 2)), np.zeros((2, 2)), names, 0) == []
    result = verifier.build_baseline_graph(
        np.array([[1, 0.9], [0, 1]]), np.array([[1, -0.2], [0, 1]]), names, 0
    )
    assert result == [{"source": "A", "target": "B", "weight": -0.2}]


@pytest.mark.parametrize(
    "estimate,signed,names,q",
    [
        (np.zeros((2, 2)), np.zeros((2, 2)), ["A", "B"], 1.1),
        (np.zeros((2, 2)), np.zeros((2, 2)), ["A", "A"], 0.5),
        (np.zeros((3, 2)), np.zeros((2, 2)), ["A", "B"], 0.5),
        (np.full((2, 2), np.inf), np.zeros((2, 2)), ["A", "B"], 0.5),
        (-np.ones((2, 2)), np.zeros((2, 2)), ["A", "B"], 0.5),
    ],
)
def test_invalid_baseline_inputs(estimate, signed, names, q):
    with pytest.raises(ValueError):
        verifier.build_baseline_graph(estimate, signed, names, q)


def test_directional_mask_matches_senn_matrix_multiplication():
    net = SENNGC(2, 1, 2, 1, torch.device("cpu"))

    class Ones(torch.nn.Module):
        def forward(self, inputs):
            return torch.ones((len(inputs), 4))

    net.coeff_nets = torch.nn.ModuleList([Ones()])
    mask = verifier.create_mask_matrix([{"source": "A", "target": "B"}], {"A": 0, "B": 1}, 2)
    predicted, _ = net(torch.tensor([[[2.0, 0.0]]]), mask=mask.T)
    assert predicted.tolist() == [[2.0, 2.0]]  # A influences B, not the reverse.
    predicted, _ = net(torch.tensor([[[0.0, 3.0]]]), mask=mask.T)
    assert predicted.tolist() == [[0.0, 3.0]]


def test_unknown_edges_and_invalid_indices_fail():
    with pytest.raises(ValueError):
        verifier.create_mask_matrix([{"source": "ghost", "target": "B"}], {"A": 0, "B": 1}, 2)
    with pytest.raises(ValueError):
        verifier.create_mask_matrix([], {"A": 0, "B": 0}, 2)


def test_dense_and_masked_models_never_fit_score_partition(monkeypatch):
    fitted, scored, masks = [], [], []

    class Dummy:
        device = torch.device("cpu")
        causal_mask = None

        def _training(self, data, **kwargs):
            fitted.append(data.copy())
            assert kwargs == {"persist": False, "calibrate": False}
            if self.causal_mask is not None:
                masks.append(self.causal_mask.clone())

        def eval(self):
            return self

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

        def _testing_step(self, data, **kwargs):
            coeffs = torch.tensor([[[[0.0, 0.0], [0.8, 0.0]]]])
            return None, None, None, coeffs

        def forward(self, data, **kwargs):
            scored.append(data.copy())
            return torch.zeros_like(torch.from_numpy(data)), torch.from_numpy(data)

    monkeypatch.setattr(verifier, "_model", lambda *a, **kw: Dummy())
    data = np.arange(200, dtype=np.float32).reshape(1, 100, 2)
    edges, weights, magnitude = verifier.get_initial_graph_from_aerca(data, ["A", "B"], 0)
    assert edges[0]["source"] == "A" and edges[0]["target"] == "B"
    assert magnitude[0, 1] == pytest.approx(0.8)
    mse, count, _, signed = verifier.run_masked_aerca(data, edges, ["A", "B"], weights)
    np.testing.assert_array_equal(fitted[0], data[:, :80])
    np.testing.assert_array_equal(fitted[1], data[:, :80])
    np.testing.assert_array_equal(scored[0], data[0, 80:])
    assert masks[0].tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert count == 1 and signed[0]["weight"] == pytest.approx(0.8)
    assert mse == pytest.approx(np.square(data[0, 80:]).mean())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dense_epochs": 0},
        {"lr": float("nan")},
        {"device": "bogus"},
        {"seed": -1},
        {"hidden_layer_size": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        verifier.VerifierConfig(**kwargs)


def test_duplicate_model_variables_fail():
    with pytest.raises(ValueError):
        verifier._model(["A", "A"], verifier.VerifierConfig())
