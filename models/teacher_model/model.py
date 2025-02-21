import torch

from ..base_model import BaseModel
import torch.nn as nn


class TeacherEvaluator(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_size = config.get('input_size', 768)
        self.hidden_size = config.get('hidden_size', 256)
        self.num_heads = config.get('num_heads', 4)

        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size * 4
            ),
            num_layers=3
        )

        self.evaluation_heads = nn.ModuleDict({
            'content_mastery': nn.Linear(self.hidden_size, 1),
            'explanation_clarity': nn.Linear(self.hidden_size, 1),
            'engagement': nn.Linear(self.hidden_size, 1),
            'structure': nn.Linear(self.hidden_size, 1)
        })

    def forward(self, features):
        projected = self.input_projection(features)
        encoded = self.transformer_layers(projected)

        return {
            name: torch.sigmoid(head(encoded.mean(dim=1)))
            for name, head in self.evaluation_heads.items()
        }