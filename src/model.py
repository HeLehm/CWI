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
        # zero out padding
        logits = logits * attention_mask.unsqueeze(-1).to(torch.float)
        # zero out cls token
        logits = logits * (input_ids != 101).unsqueeze(-1).to(torch.float)
        return logits