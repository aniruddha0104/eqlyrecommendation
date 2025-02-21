import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_config(self):
        return self.config

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']