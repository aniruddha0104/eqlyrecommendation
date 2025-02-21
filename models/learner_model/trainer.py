import torch
import torch.nn as nn
from typing import Dict, Any
from torch.utils.data import DataLoader


class LearnerTrainer:
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        self.criterion = nn.MSELoss()

    def train_step(self, batch, teacher_outputs):
        self.model.train()
        _, labels = batch
        outputs = self.model(teacher_outputs.detach())
        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}