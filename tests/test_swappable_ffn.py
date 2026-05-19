import pytest

torch = pytest.importorskip("torch")

from moe_transformer_lab.config import ModelConfig
from moe_transformer_lab.trainable import (
    DecoderOnlyTransformer,
    DenseMLP,
    MoEFeedForward,
    TransformerBlock,
    replace_feed_forward_modules,
)


def tiny_config(ffn_type: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        max_seq_len=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        d_ff=32,
        ffn_type=ffn_type,
        n_experts=4,
        top_k=1,
    )


def test_dense_and_moe_follow_same_feed_forward_contract():
    x = torch.randn(2, 5, 16)
    dense = DenseMLP(tiny_config("dense"))
    moe = MoEFeedForward(tiny_config("moe"))

    dense_y, dense_aux = dense(x)
    moe_y, moe_aux = moe(x)

    assert dense_y.shape == x.shape
    assert moe_y.shape == x.shape
    assert dense_aux.item() == 0.0
    assert moe_aux.item() >= 0.0


def test_transformer_block_accepts_dense_or_moe_without_shape_change():
    x = torch.randn(2, 5, 16)
    dense_block = TransformerBlock(tiny_config("dense"))
    moe_block = TransformerBlock(tiny_config("moe"))

    dense_y, dense_aux = dense_block(x)
    moe_y, moe_aux = moe_block(x)

    assert dense_y.shape == x.shape
    assert moe_y.shape == x.shape
    assert dense_aux.item() == 0.0
    assert moe_aux.item() >= 0.0


def test_replacing_feed_forward_preserves_attention_modules():
    model = DecoderOnlyTransformer(tiny_config("dense"))
    attention_ids = [id(block.attention) for block in model.blocks]

    replace_feed_forward_modules(model, "moe")

    assert model.config.ffn_type == "moe"
    assert [id(block.attention) for block in model.blocks] == attention_ids
    assert all(isinstance(block.feed_forward, MoEFeedForward) for block in model.blocks)


def test_lm_forward_works_for_dense_and_moe():
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))

    for ffn_type in ["dense", "moe"]:
        model = DecoderOnlyTransformer(tiny_config(ffn_type))
        out = model(idx, targets)
        assert out.logits.shape == (2, 8, 64)
        assert out.loss.ndim == 0
        assert out.aux_loss.item() >= 0.0
