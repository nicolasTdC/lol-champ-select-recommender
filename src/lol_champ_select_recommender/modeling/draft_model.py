from __future__ import annotations


class MissingTorchError(RuntimeError):
    pass


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise MissingTorchError(
            "PyTorch is not installed. Install it in a Python 3.11/3.12 environment with: "
            "python -m pip install torch"
        ) from exc
    return torch, nn


def build_model_class():
    torch, nn = require_torch()

    class SharedFeatureDraftTransformer(nn.Module):
        def __init__(
            self,
            *,
            shared_vocab_size: int,
            champion_vocab_size: int,
            d_model: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(shared_vocab_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_norm = nn.LayerNorm(d_model)
            self.output = nn.Linear(d_model, champion_vocab_size)

        def forward(self, feature_ids, query_index):
            embedded = self.embedding(feature_ids).sum(dim=2)
            encoded = self.encoder(embedded)
            batch_index = torch.arange(encoded.size(0), device=encoded.device)
            query_hidden = encoded[batch_index, query_index]
            return self.output(self.output_norm(query_hidden))

    return SharedFeatureDraftTransformer
