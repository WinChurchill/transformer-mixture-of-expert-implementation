import pytest

torch = pytest.importorskip("torch")

from moe_transformer_lab.manual import MultiHeadSelfAttention, causal_mask


def test_multi_head_attention_shape_and_backward():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(d_model=8, n_heads=2)
    x = torch.randn(2, 4, 8)
    y = attn.forward(x, causal_mask(4))
    dx = attn.backward(torch.ones_like(y))

    assert y.shape == x.shape
    assert dx.shape == x.shape
    for param in attn.parameters():
        assert param.grad.shape == param.data.shape


def test_causal_attention_future_token_does_not_change_past_output():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(d_model=8, n_heads=2)
    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()
    x2[:, 3, :] = torch.randn(8) * 100.0

    y1 = attn.forward(x1, causal_mask(4))
    y2 = attn.forward(x2, causal_mask(4))

    assert torch.allclose(y1[:, :3, :], y2[:, :3, :], atol=1e-5)
