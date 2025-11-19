import torch
import pytest

from models.model_MLP import ResidualBlock, LandmarkMLP


def test_residual_block_forward_shape():
    dim = 64
    block = ResidualBlock(dim=dim, expansion=2, dropout=0.1)
    x = torch.randn(8, dim)
    out = block(x)

    assert out.shape == (8, dim)

    assert not torch.allclose(out, x)


def test_residual_block_backward():
    dim = 64
    block = ResidualBlock(dim=dim)
    x = torch.randn(4, dim, requires_grad=True)

    out = block(x).sum()
    out.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_landmark_mlp_output_shape():
    batch = 16
    input_dim = 63
    num_classes = 24

    model = LandmarkMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=3,
        dropout=0.1,
    )

    x = torch.randn(batch, input_dim)
    logits = model(x)

    assert logits.shape == (batch, num_classes)
    assert torch.isfinite(logits).all()


def test_landmark_mlp_forward_gradients():
    model = LandmarkMLP(input_dim=63, num_classes=24)
    x = torch.randn(5, 63, requires_grad=True)

    out = model(x).sum()
    out.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_landmark_mlp_parameter_count():
    model = LandmarkMLP(
        input_dim=63,
        num_classes=24,
        hidden_dim=256,
        num_blocks=2,
        dropout=0.3,
    )

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 50_000
    assert total_params < 1_000_000 


def test_residual_block_actually_residual():
    dim = 32
    block = ResidualBlock(dim=dim)
    x = torch.randn(1, dim)
    out = block(x)

    diff = (out - x).abs().sum().item()
    assert diff > 1e-6
