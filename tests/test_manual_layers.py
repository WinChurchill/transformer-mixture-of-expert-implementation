import pytest

torch = pytest.importorskip("torch")

from moe_transformer_lab.manual import LayerNorm, Linear, SoftmaxCrossEntropy, causal_mask


def zero_params(module):
    for param in module.parameters():
        param.zero_grad()


def test_linear_backward_matches_finite_difference():
    torch.manual_seed(0)
    layer = Linear(3, 2, dtype=torch.float64)
    x = torch.randn(4, 3, dtype=torch.float64)

    y = layer.forward(x)
    loss = (y * y).sum()
    layer.backward(2.0 * y)

    eps = 1e-6
    idx = (1, 0)
    original = layer.weight.data[idx].item()
    layer.weight.data[idx] = original + eps
    loss_pos = (layer.forward(x) ** 2).sum()
    layer.weight.data[idx] = original - eps
    loss_neg = (layer.forward(x) ** 2).sum()
    layer.weight.data[idx] = original
    numeric = (loss_pos - loss_neg) / (2.0 * eps)

    assert torch.allclose(layer.weight.grad[idx], numeric, rtol=1e-5, atol=1e-5)
    assert loss.item() > 0


def test_layer_norm_backward_matches_autograd():
    torch.manual_seed(0)
    manual = LayerNorm(4, dtype=torch.float64)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    dout = torch.randn_like(x)

    y = manual.forward(x)
    dx = manual.backward(dout)

    x_ref = x.detach().clone().requires_grad_(True)
    gamma = manual.gamma.data.detach().clone().requires_grad_(True)
    beta = manual.beta.data.detach().clone().requires_grad_(True)
    y_ref = torch.nn.functional.layer_norm(x_ref, (4,), gamma, beta, eps=manual.eps)
    y_ref.backward(dout)

    assert torch.allclose(y, y_ref.detach(), atol=1e-10)
    assert torch.allclose(dx, x_ref.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(manual.gamma.grad, gamma.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(manual.beta.grad, beta.grad, rtol=1e-5, atol=1e-6)


def test_softmax_cross_entropy_backward_rows_sum_to_zero():
    torch.manual_seed(0)
    loss_fn = SoftmaxCrossEntropy()
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[0, 1, 2], [2, 3, 4]])
    loss = loss_fn.forward(logits, targets)
    grad = loss_fn.backward().reshape(2, 3, 5)

    assert loss.ndim == 0
    assert torch.allclose(grad.sum(dim=-1), torch.zeros(2, 3), atol=1e-6)


def test_causal_mask_is_lower_triangular():
    mask = causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)
