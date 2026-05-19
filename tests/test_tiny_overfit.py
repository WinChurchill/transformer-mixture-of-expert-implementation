import pytest

torch = pytest.importorskip("torch")

from moe_transformer_lab.config import ModelConfig
from moe_transformer_lab.trainable import DecoderOnlyTransformer


def config(ffn_type: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
        ffn_type=ffn_type,
        n_experts=2,
        top_k=1,
        aux_loss_coef=0.001,
    )


@pytest.mark.parametrize("ffn_type", ["dense", "moe"])
def test_tiny_batch_loss_decreases(ffn_type):
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(config(ffn_type))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]) % 16
    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]]) % 16

    with torch.no_grad():
        initial = model(idx, targets).loss.item()

    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss = model(idx, targets).loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = model(idx, targets).loss.item()

    assert final < initial
