from __future__ import annotations

from ..roles import POSITION_ORDER


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
            coarse_bucket_size: int = 0,
            d_model: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            use_role_heads: bool = True,
            use_hierarchy: bool = False,
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
            self.use_role_heads = use_role_heads
            self.use_hierarchy = use_hierarchy and coarse_bucket_size > 0
            if self.use_hierarchy:
                self.role_outputs = nn.ModuleDict(
                    {role: nn.Linear(d_model, champion_vocab_size) for role in POSITION_ORDER}
                )
                self.role_coarse_outputs = nn.ModuleDict(
                    {role: nn.Linear(d_model, coarse_bucket_size) for role in POSITION_ORDER}
                )
            else:
                self.role_outputs = nn.ModuleDict(
                    {role: nn.Linear(d_model, champion_vocab_size) for role in POSITION_ORDER}
                )
                self.role_coarse_outputs = None

        def forward(
            self,
            feature_ids,
            query_index,
            role_index=None,
            target_coarse_index=None,
        ):
            embedded = self.embedding(feature_ids).sum(dim=2)
            encoded = self.encoder(embedded)
            batch_index = torch.arange(encoded.size(0), device=encoded.device)
            query_hidden = encoded[batch_index, query_index]
            hidden = self.output_norm(query_hidden)
            if self.use_hierarchy:
                if role_index is None:
                    raise ValueError("role_index is required when hierarchy is enabled")
                champion_logits = []
                coarse_logits = []
                for row_index, role_bucket_index in enumerate(role_index.tolist()):
                    role = POSITION_ORDER[int(role_bucket_index)]
                    role_coarse_logits = self.role_coarse_outputs[role](hidden[row_index])
                    coarse_logits.append(role_coarse_logits)
                    champion_logits.append(self.role_outputs[role](hidden[row_index]))
                return torch.stack(champion_logits, dim=0), torch.stack(coarse_logits, dim=0)

            if not self.use_role_heads:
                return self.output(hidden)

            if role_index is None:
                raise ValueError("role_index is required when role heads are enabled")
            role_logits = []
            for row_index, role_bucket_index in enumerate(role_index.tolist()):
                role = POSITION_ORDER[int(role_bucket_index)]
                role_logits.append(self.role_outputs[role](hidden[row_index]))
            return torch.stack(role_logits, dim=0)

    return SharedFeatureDraftTransformer
