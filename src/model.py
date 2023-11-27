import torch
from torch import nn


class ModelForTokenRegression(nn.Module):
    def __init__(self, backbone, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone= backbone
        self.regression = nn.Linear(backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.regression(sequence_output)
        return logits