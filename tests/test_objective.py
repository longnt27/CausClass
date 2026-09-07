import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

from utils.aerca_utils import compute_kl_divergence


@pytest.mark.parametrize("dimension", [1, 2, 6])
def test_gaussian_kl_matches_torch(dimension):
    generator = torch.Generator().manual_seed(42)
    samples = torch.randn(50, dimension, dtype=torch.float64, generator=generator)
    samples = (samples * 1.7 + 0.3).requires_grad_(True)
    covariance = torch.atleast_2d(torch.cov(samples.T))
    covariance = covariance + torch.finfo(samples.dtype).eps * torch.eye(
        dimension, dtype=samples.dtype
    )
    expected = kl_divergence(
        MultivariateNormal(samples.mean(0), covariance),
        MultivariateNormal(
            torch.zeros(dimension, dtype=samples.dtype), torch.eye(dimension, dtype=samples.dtype)
        ),
    )
    actual = compute_kl_divergence(samples, torch.device("cpu"))
    assert actual.item() == pytest.approx(expected.item(), abs=1e-10)
    assert actual >= 0
    actual.backward()
    assert torch.isfinite(samples.grad).all()


def test_singular_covariance_remains_finite():
    assert torch.isfinite(compute_kl_divergence(torch.ones(10, 2), torch.device("cpu")))


@pytest.mark.parametrize(
    "samples", [torch.ones(1, 2), torch.ones(2), torch.full((4, 2), float("nan"))]
)
def test_invalid_kl_data(samples):
    with pytest.raises(ValueError):
        compute_kl_divergence(samples, torch.device("cpu"))
