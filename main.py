import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel

from src.data import load_pd, ComplexWordDataset
from src.model import ModelForTokenRegression



def train(model, train_loader, dev_loader, optimizer, device, num_epochs=3):
    # Loss function
    criterion = torch.nn.MSELoss()  # or another appropriate loss function

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        samples_seen = 0

        for batch in (pbar := tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            # Move batch to device
            token_ids = batch['token_ids'].to(device)
            label_probs = batch['label_probs'].to(device).unsqueeze(-1)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(token_ids, attention_mask=attention_mask)
            loss = criterion(outputs, label_probs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            samples_seen += len(batch)
            pbar.set_postfix({'loss': total_loss / samples_seen})

        print(f"Epoch {epoch + 1}: Average Training Loss = {total_loss / samples_seen}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        total_val_seen = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Validating Epoch {epoch + 1}"):
                token_ids = batch['token_ids'].to(device)
                label_probs = batch['label_probs'].to(device).unsqueeze(-1)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(token_ids, attention_mask=attention_mask)
                loss = criterion(outputs, label_probs)
                total_val_loss += loss.item()
                total_val_seen += len(batch)

        print(f"Epoch {epoch + 1}: Average Validation Loss = {total_val_loss / total_val_seen}")


def main(
    backbone_name = 'bert-base-uncased',
    batch_size = 8,
    device = "mps",
    finetune="head",
    lr=1e-5,
):
    # model stuff
    tokenizer = BertTokenizerFast.from_pretrained(backbone_name)
    backbone = BertModel.from_pretrained(backbone_name)
    model = ModelForTokenRegression(backbone).to(device)

    parameters = None
    if finetune == "head":
        parameters = model.regression.parameters()
        # turn off grad for the backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif finetune == "all":
        parameters = model.parameters()
    else:
        raise ValueError(f"finetune must be 'head' or 'all', got {finetune}")


    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW(parameters, lr=lr)

    # data stuff
    t,d = load_pd()
    train_ds = ComplexWordDataset(t, tokenizer)
    dev_ds = ComplexWordDataset(d, tokenizer)

    train_dl = train_ds.get_dataloader(batch_size=batch_size, shuffle=True)
    dev_dl = dev_ds.get_dataloader(batch_size=batch_size, shuffle=False)

    train(model, train_dl, dev_dl, optimizer, device)






if __name__ == "__main__":
    main()