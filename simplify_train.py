import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from sklearn.model_selection import train_test_split
from src.cwi.model import ModelForTokenRegression
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

import adapters

nltk.download('punkt')

from src.simplify.dataset import ParaphraseDataset, get_chatgpt_dataset_pd

def main(backbone_name, device, batch_size, lr, epochs, cwi_model_path, only_source="cnn_news", cwi_multiplier=1.0):

    cwi_model = ModelForTokenRegression.load(cwi_model_path).to(device)

    assert cwi_model.backbone_name == backbone_name


    df = get_chatgpt_dataset_pd()

    if only_source is not None:
        print(f"Filtering only {only_source} source")
        print(f"Before: {len(df)}")
        df = df[df.source == only_source]
        print(f"After: {len(df)}")
    

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(backbone_name)
    adapters.init(model)
    model.add_adapter("simplify")
    model.train_adapter("simplify")
    
    model.to(device)

    
    # Prepare the dataset and dataloader
    train_dataset = ParaphraseDataset(tokenizer, train_df, cache=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ParaphraseDataset(tokenizer, val_df, cache=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer
    paramerters_to_optimize = [param for param in model.parameters() if param.requires_grad]
    print(f"Number of parameters to optimize: {sum([param.numel() for param in paramerters_to_optimize])}")
    print(f"Number of parameters in model: {sum([param.numel() for param in model.parameters()])}")
    optimizer = AdamW(paramerters_to_optimize, lr=lr)
    
    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_epoch(model, train_loader, optimizer, cwi_model, cwi_multiplier=cwi_multiplier)
        avg_bleu_score = calculate_bleu_score(model, val_loader, tokenizer, device)
        print(f"Average BLEU Score on Validation Set: {avg_bleu_score}")

    # Save the model
    save_dir = "./models/simplify"
    save_dir = os.path.join(save_dir, backbone_name + f"_simplify_{lr}_{batch_size}_{epochs}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_adapter(save_dir, "simplify")


# Evaluation using BLEU Score
def calculate_bleu_score(model, dataloader, tokenizer, device):
    model.eval()
    bleu_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)

            # Generate paraphrases
            generated_ids = model.generate(input_ids, attention_mask=attention_mask)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Calculate BLEU for each paraphrase
            for i, gen_text in enumerate(generated_texts):
                references = [word_tokenize(ref) for ref in batch['paraphrases'][i]]
                candidate = word_tokenize(gen_text)
                score = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))  # Adjust weights as needed
                bleu_scores.append(score)

    # Calculate the average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score


def train_epoch(model, train_loader, optimizer, cwi_model, cwi_multiplier=1.0):
    # Training Loop
    model.train()
    cwi_model.eval()

    for batch in (pbar := tqdm(train_loader)):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].squeeze(1).to(model.device)
        attention_mask = batch['attention_mask'].squeeze(1).to(model.device)
        labels = batch['labels'].squeeze(1).to(model.device)

        # caulate paraphrase loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # caulate CWI loss
        cwi_inputs = outputs.logits.argmax(dim=-1)
        # craet attention mask based in special tokens
        cwi_attention_mask = (cwi_inputs != model.config.pad_token_id).to(torch.long)
        cwi_logits = cwi_model(cwi_inputs, cwi_attention_mask).squeeze(-1)
        # camp 0, 1
        cwi_logits = torch.clamp(cwi_logits, 0, 1)
        # calculate loss (mae)
        cwi_loss = (cwi_logits * cwi_attention_mask).mean() * cwi_multiplier

        para_loss = loss.item()
        cwi_loss = cwi_loss.item()

        loss += cwi_loss

        loss.backward()
        optimizer.step()
        pbar.set_description(f"Train Loss: {loss.item():.3f}, Para Loss: {para_loss:.3f}, CWI Loss: {cwi_loss:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='humarin/chatgpt_paraphraser_on_T5_base')
    parser.add_argument('--cwi_multiplier', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--cwi_model_path', type=str, default='./models/cwi/humarin/chatgpt_paraphraser_on_T5_base_adapter_0.001_10_False')
    parser.add_argument('--only_source', type=str, default=None, choices=["cnn_news", "quora", "squad_2", None])
    args = parser.parse_args()
    main(args.backbone, args.device, args.batch_size, args.lr, args.epochs, args.cwi_model_path, args.only_source, args.cwi_multiplier)
