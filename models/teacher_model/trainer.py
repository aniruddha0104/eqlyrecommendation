import torch
import torch.nn as nn
from typing import Dict, Any
from torch.utils.data import DataLoader


class TeacherTrainer:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        self.criterion = nn.MSELoss()

    def train_step(self, batch):
        self.model.train()
        features, labels = batch
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}