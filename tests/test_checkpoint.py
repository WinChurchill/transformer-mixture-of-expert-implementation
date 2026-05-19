import pytest

torch = pytest.importorskip("torch")

from moe_transformer_lab.config import ModelConfig
from moe_transformer_lab.trainable import DecoderOnlyTransformer
from moe_transformer_lab.trainable.train_utils import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        ffn_type="moe",
        n_experts=2,
        top_k=1,
    )
    model = DecoderOnlyTransformer(config)
    model.eval()
    idx = torch.randint(0, 32, (2, 8))
    logits_before = model(idx).logits.detach()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=None, step=7)
    payload = load_checkpoint(path)

    restored = DecoderOnlyTransformer(ModelConfig(**payload["config"]))
    restored.load_state_dict(payload["model_state"])
    restored.eval()
    logits_after = restored(idx).logits.detach()

    assert payload["step"] == 7
    assert torch.allclose(logits_before, logits_after)
