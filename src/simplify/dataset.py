import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from tqdm import tqdm
from datasets import load_dataset



class ParaphraseDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128, para_idx=0, cache=True):
        self.para_idx = para_idx
        self.cache = cache
        if cache:
            self._cache_tokenized_data(data, tokenizer, max_length)
        else:
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.para_idx = para_idx


    def _cache_tokenized_data(self, data, tokenizer, max_length):
        self.tokenized_data = []
        for idx, row in tqdm(data.iterrows(), desc='Tokenizing dataset', total=len(data)):
            original_sentence = row['text']
            paraphrase = row['paraphrases'][self.para_idx]
            inputs = tokenizer.encode_plus(
                f'paraphrase: {original_sentence}', 
                add_special_tokens=True, 
                max_length=max_length, 
                padding="max_length", 
                return_tensors="pt", 
                truncation=True
            )

            labels = tokenizer.encode_plus(
                paraphrase, 
                add_special_tokens=True, 
                max_length=max_length, 
                padding="max_length", 
                return_tensors="pt", 
                truncation=True
            )
            self.tokenized_data.append({
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten(),
                'labels': labels['input_ids'].flatten(),
                'paraphrases': row['paraphrases']  # For evaluation
            })

    def __len__(self):
        if self.cache:
            return len(self.tokenized_data)
        return len(self.data)

    def __getitem__(self, idx):
        if self.cache:
            return self.tokenized_data[idx]
        
        row = self.data.iloc[idx]
        original_sentence = row['text']
        paraphrase = row['paraphrases'][self.para_idx]
        inputs = self.tokenizer.encode_plus(
            f'paraphrase: {original_sentence}', 
            add_special_tokens=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt", 
            truncation=True
        )

        labels = self.tokenizer.encode_plus(
            paraphrase, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt", 
            truncation=True
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten(),
            'paraphrases': row['paraphrases']  # For evaluation
        }


def get_chatgpt_dataset_pd():
    # 'humarin/chatgpt-paraphrases'
    dataset = load_dataset('humarin/chatgpt-paraphrases', split='train')
    df = pd.DataFrame(dataset)
    df['paraphrases'] = df['paraphrases'].apply(eval)
    return df

if __name__ == '__main__':
    df = get_chatgpt_dataset_pd()
    print(df.head())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('humarin/chatgpt_paraphraser_on_T5_base')
    dataset = ParaphraseDataset(tokenizer, data=df)
    print(dataset[0])



    
