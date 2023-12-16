import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5EncoderModel
import adapters

class ModelForTokenRegression(nn.Module):
    def __init__(self, backbone_name, finetune='head', *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert finetune in ['head', 'adapter']

        self.backbone_name = backbone_name
        self.load_backbone(backbone_name, finetune)
        
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        self.regression = nn.Linear(self.backbone.config.hidden_size, 1)

        self.special_ids = tokenizer.all_special_ids

        if finetune == "head":
            self.freeze_backbone()

    def load_backbone(self, backbone_name, finetune):
        # Load the configuration of the model
        config = AutoConfig.from_pretrained(backbone_name)

        if config.model_type == "t5":
            self.backbone = T5EncoderModel.from_pretrained(backbone_name)
        else:
            self.backbone = AutoModel.from_pretrained(backbone_name)

        if finetune == "adapter":
            adapters.init(self.backbone)
            self.backbone.add_adapter("complex_word")
            self.backbone.train_adapter("complex_word")


    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.regression(sequence_output)
        logits = logits * attention_mask.unsqueeze(-1).to(torch.float)

        # mask special tokens
        for special_id in self.special_ids:
            logits = logits * (input_ids != special_id).unsqueeze(-1).to(torch.float)
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
            # this is the case if the model doen't have an adapter named 'complex_word'
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
            adapters.init(model.backbone)
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

if __name__ == "__main__":
    # test for T5
    model = ModelForTokenRegression('humarin/chatgpt_paraphraser_on_T5_base', finetune='adapter')
    print("init model done")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save(tmpdirname)
        print("save model done")
        model = ModelForTokenRegression.load(tmpdirname)
        print("load model done")