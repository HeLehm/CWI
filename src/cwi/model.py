import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ModelForTokenRegression(nn.Module):
    def __init__(self, backbone_name, finetune='head', *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert finetune in ['head', 'adapter']

        self.backbone_name = backbone_name

        self.backbone = AutoModel.from_pretrained(backbone_name)
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        if finetune == "adapter":
            self.backbone.add_adapter(
                "complex_word",
                config="pfeiffer",
            )
            self.backbone.train_adapter("complex_word")
            self.backbone.set_active_adapters("complex_word")

        self.regression = nn.Linear(self.backbone.config.hidden_size, 1)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        if finetune == "head":
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.regression(sequence_output)
        logits = logits * attention_mask.unsqueeze(-1).to(torch.float)
        logits = logits * (input_ids != self.cls_id).unsqueeze(-1).to(torch.float)
        logits = logits * (input_ids != self.sep_id).unsqueeze(-1).to(torch.float)
        return logits

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        adapter_dir_path = os.path.join(dir_path, 'adapter')
        os.makedirs(adapter_dir_path, exist_ok=True)

        state_dict_path = os.path.join(dir_path, 'state_dict.pt')
        torch.save({
            'backbone_name': self.backbone.config._name_or_path,
            'regression_state_dict': self.regression.state_dict(),
        }, state_dict_path)

        # save adapter if exists
        try:
            self.backbone.save_adapter(adapter_dir_path, 'complex_word')
        except ValueError:
            # this is the case if teh model doen't have an adapter named 'complex_word'
            pass

    @classmethod
    def load(cls, dir_path, device='cpu'):
        file_path = os.path.join(dir_path, 'state_dict.pt')
        checkpoint = torch.load(file_path, map_location=device)

        # Create a new model instance
        backbone_name = checkpoint['backbone_name']
        # act like we are loading a model that was finetuned on the head
        model = cls(backbone_name, finetune='head')

        # Load the regression layer state dict
        model.regression.load_state_dict(checkpoint['regression_state_dict'])

        # Load the adapter state dict if it exists
        # just try
        try:
            adpater_name = model.backbone.load_adapter(os.path.join(dir_path, 'adapter'))
            print(f"Adapter with name {adpater_name} loaded.")
            # set active
            model.backbone.set_active_adapters(adpater_name)
        except Exception as e:
            print(e)
            # this is the case if teh model doen't have an adapter named 'complex_word'
            pass

        return model


def predict_batch(model, batch, device, binary=True):
    if binary:
        label_probs = batch['label_binary'].to(device).unsqueeze(-1).to(torch.float)
    else:
        label_probs = batch['label_probs'].to(device).unsqueeze(-1)
    token_ids = batch['token_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    outputs = model(token_ids, attention_mask=attention_mask)
    return outputs, label_probs, token_ids