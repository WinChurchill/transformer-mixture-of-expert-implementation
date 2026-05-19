import pytest

from moe_transformer_lab.config import ModelConfig


def test_config_validation_accepts_dense_and_moe():
    ModelConfig(ffn_type="dense").validate()
    ModelConfig(ffn_type="moe", n_experts=4, top_k=2).validate()


def test_config_validation_rejects_bad_top_k():
    with pytest.raises(ValueError):
        ModelConfig(ffn_type="moe", n_experts=2, top_k=3).validate()
