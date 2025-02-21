import torch

from ..base_model import BaseModel
import torch.nn as nn


class LearnerModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_size = config.get('input_size', 768)
        self.hidden_size = config.get('hidden_size', 256)

        self.understanding_network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 3)
        )

    def forward(self, teacher_features):
        understanding_scores = torch.sigmoid(self.understanding_network(teacher_features))
        return {
            'concept_grasp': understanding_scores[:, 0],
            'knowledge_retention': understanding_scores[:, 1],
            'application_ability': understanding_scores[:, 2]
        }