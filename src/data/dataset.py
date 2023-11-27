import torch
from torch.utils.data import Dataset, DataLoader
from .utils import convert_dataframe, collate

# Preprocess the data
class ComplexWordDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = convert_dataframe(dataframe, tokenizer, max_len)
        self.len = len(self.data)

    def __getitem__(self, index):
        # token_ids and label_probs and attention_mask
        data = self.data.iloc[index]
        return {
            'token_ids': torch.tensor(data.token_ids, dtype=torch.long),
            'label_probs': data.label_probs.to(torch.float),
            'attention_mask': torch.tensor(data.attention_mask, dtype=torch.long),
        }

    def __len__(self):
        return self.len
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=collate, **kwargs)