import torch
from torch.utils.data import Dataset
from .utils import convert_dataframe

# Preprocess the data
class ComplexWordDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = convert_dataframe(dataframe, tokenizer, max_len)
        self.len = len(self.data)

    def __getitem__(self, index):
        # token_ids and label_probs and attention_mask
        data = self.data.iloc[index]
        return {
            'token_ids': data.token_ids.to(torch.long),
            'label_probs': data.label_probs.to(torch.float),
            'attention_mask': data.attention_maskto(torch.long),
        }

    def __len__(self):
        return self.len
    

