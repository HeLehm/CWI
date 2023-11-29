import torch
from torch import nn


class ModelForTokenRegression(nn.Module):
    def __init__(self, backbone, tokenizer,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone= backbone
        self.regression = nn.Linear(backbone.config.hidden_size, 1)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # shape (batch_size, seq_len, hidden_size)
        logits = self.regression(sequence_output) # shape (batch_size, seq_len, 1)
        # apply sigmoid
        #logits = torch.sigmoid(logits)
        # zero out padding
        logits = logits * attention_mask.unsqueeze(-1).to(torch.float)
        # zero out cls token
        logits = logits * (input_ids != self.cls_id).unsqueeze(-1).to(torch.float)
        # zero out sep token
        logits = logits * (input_ids != self.sep_id).unsqueeze(-1).to(torch.float)
        return logits
    
def predict_batch(model, batch, device, binary=True):
    if binary:
        label_probs = batch['label_binary'].to(device).unsqueeze(-1).to(torch.float)
    else:
        label_probs = batch['label_probs'].to(device).unsqueeze(-1)
    token_ids = batch['token_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    outputs = model(token_ids, attention_mask=attention_mask)
    return outputs, label_probs, token_ids