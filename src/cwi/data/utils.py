import pandas as pd
import torch
import numpy as np
import os

COLUMNS = [
    "HITID", # sentence id
    "text", # sentence
    'start_index', # start index of the target word
    'end_index', # end index of the target word
    'target_word', # target word
    'num_native_asked', # is just always 10
    'num_nonnative_asked', # is just always 10
    'native_marked', # num native speaker's that marked as complex 
    'nonnative_marked', # nonnative speaker's that marked as complex
    'label_binary', # binary label (is complex or not)
    'label_prob', # probability label (higher number means more complex) (<the number of annotators who marked the word as difficult>/<the total number of annotators>.)
]


def get_paths():
    """
    returns the paths to the train and dev data
    """
    this_file_path = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file_path)
    this_dir = os.path.dirname(this_dir)
    src_dir = os.path.dirname(this_dir)
    root_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(root_dir, 'Assignment-Option-3-cwi-datta')
    traindev_dir = os.path.join(data_dir, 'traindev')
    return os.path.join(traindev_dir, 'News_Train.tsv'), os.path.join(traindev_dir, 'News_Dev.tsv')


def load_pd():
    """
    loads the data as a pandas dataframe
    """
    # load data
    path_train, path_dev = get_paths()

    train = pd.read_csv(path_train, sep='\t', names=COLUMNS)
    dev = pd.read_csv(path_dev, sep='\t', names=COLUMNS)

    return train, dev


def convert_dataframe(df, tokenizer, max_len=512):
    """
    Takes a daframe and converts it to a new dataframe with the following columns:
    - HITID
    - text
    - token_ids
    - label_probs
    - attention_mask
    """
    # gte unique sentences (col: text)
    sentences = df.text.unique()

    # create a new dataframe
    df_new = []

    for _, sentence in enumerate(sentences):
        # get the row of the sentence
        rows = df[df.text == sentence].copy()
        # sort rows by ength of target word
        # shortest last
        rows['target_length'] = rows['target_word'].apply(len)
        rows = rows.sort_values(by='target_length', ascending=False)

        # tokinize the sentence
        encoding = tokenizer(
            sentence,
            return_offsets_mapping=True,
            max_length=max_len,
            padding='max_length',
        )

        target_tensor = torch.zeros(len(encoding['input_ids']))
        for _, r in rows.iterrows():
            if r.label_prob == 0:
                continue
            start_token_index = None
            end_token_index = None
            # find relevant tokens via r.start_index and r.end_index
            for token_index, offset in enumerate(encoding['offset_mapping']):
                # multiple tokens can be affected
                
                if start_token_index is None and offset[0] <= r.start_index and r.start_index < offset[1]:
                    start_token_index = token_index
                end_token_index = token_index 

                # break condition ( tokenindex is too big)
                if offset[1] >= r.end_index:
                    break

            # asssert not None
            assert start_token_index is not None and end_token_index is not None

            # set the target tensor
            target_tensor[start_token_index:end_token_index + 1] = r.label_prob
                
        # append to the new dataframe
        df_new.append({
            'HITID': rows.iloc[0].HITID,
            'text': sentence,
            'token_ids': encoding['input_ids'],
            'label_probs': target_tensor,
            'attention_mask': encoding['attention_mask'],
        })

    df_new = pd.DataFrame(df_new)

    return df_new


def interpolate_color(label_prob):
    # Interpolate between white (255, 255, 255) and red (255, 0, 0)
    r = 255
    g = int(255 * (1 - label_prob))
    b = int(255 * (1 - label_prob))
    return r, g, b


def collate(batch):
    """
    collate function for the dataloader
    """
    # get the max length of the token_ids
    max_len = max([len(x['token_ids']) for x in batch])
    # create the tensors
    token_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    label_probs = torch.zeros((len(batch), max_len), dtype=torch.float)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    label_binary = torch.zeros((len(batch), max_len), dtype=torch.long)
    # fill the tensors
    for i, data in enumerate(batch):
        token_ids[i, :len(data['token_ids'])] = data['token_ids']
        label_probs[i, :len(data['label_probs'])] = data['label_probs']
        attention_mask[i, :len(data['attention_mask'])] = data['attention_mask']
        label_binary[i, :len(data['label_binary'])] = data['label_binary']
    return {
        'token_ids': token_ids,
        'label_probs': label_probs,
        'attention_mask': attention_mask,
        'label_binary': label_binary,
    }

def find_stopwords(df):
    """
    find words that are never marked as complex
    looks at target_words with prob = 0 or words that are never marked as complex
    """
    complex_words = set()
    for _, row in df.iterrows():
        if row.label_prob != 0:
            complex_words.add(row.target_word)
    
    stopwords = set()
    unique_senetcnes = df.text.unique()
    for sentence in unique_senetcnes:
        words = sentence.split()
        for word in words:
            if word not in complex_words:
                stopwords.add(word)

    return stopwords