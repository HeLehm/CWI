import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.data import load_pd, ComplexWordDataset
from src.model import ModelForTokenRegression
from src.utils import depict_sample


def main(
    backbone_name = 'bert-base-uncased',
    batch_size = 8,
    device = "mps",
    finetune="adapter",
    lr=1e-5,
    num_epochs=10,
    binary=False,
):
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    
    # model stuff
    backbone = AutoModel.from_pretrained(backbone_name, add_pooling_layer=False)
    if finetune == "adapter":
        backbone.add_adapter(
            "complex_word",
            config="pfeiffer",
        )
        backbone.train_adapter("complex_word")
        backbone.set_active_adapters("complex_word")
    
    
    model = ModelForTokenRegression(backbone, tokenizer).to(device)

    parameters = None
    if finetune == "head":
        parameters = model.regression.parameters()
        # turn off grad for the backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif finetune == "all":
        parameters = model.parameters()
    elif finetune == "adapter":
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

    train(
        model,
        train_dl,
        dev_dl,
        optimizer,
        device,
        num_epochs=num_epochs,
        tokenizer=tokenizer,
        binary=binary
    )


def train(model, train_loader, dev_loader, optimizer, device, num_epochs=3, tokenizer=None, binary=True):
    # Loss function
    if binary:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        samples_seen = 0

        for batch in (pbar := tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            outputs, label_probs, _ = predict_batch(model, batch, device, binary=binary)
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
        print("EVAL SAMPLES: \n")
        with torch.no_grad():
            for batch in dev_loader:
                outputs, label_probs, token_ids = predict_batch(model, batch, device, binary=binary)

                loss = criterion(outputs, label_probs)
                total_val_loss += loss.item()
                total_val_seen += len(batch)

                
                print("Predicted:")
                depict_sample(token_ids[0], outputs[0], tokenizer)
                print("Label:")
                depict_sample(token_ids[0], label_probs[0], tokenizer)
                print()
        print()
        print(f"Epoch {epoch + 1}: Average Validation Loss = {total_val_loss / total_val_seen}")
        print()

        print("Random train sample:")
        batch = next(iter(train_loader))
        outputs, label_probs, token_ids = predict_batch(model, batch, device, binary=binary)
        print("Predicted:")
        depict_sample(token_ids[0], outputs[0], tokenizer)
        print("Label:")
        depict_sample(token_ids[0], label_probs[0], tokenizer)
        print()


def predict_batch(model, batch, device, binary=True):
    if binary:
        label_probs = batch['label_binary'].to(device).unsqueeze(-1).to(torch.float)
    else:
        label_probs = batch['label_probs'].to(device).unsqueeze(-1)
    token_ids = batch['token_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    outputs = model(token_ids, attention_mask=attention_mask)
    return outputs, label_probs, token_ids






def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--finetune", type=str, default="adapter")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--binary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        backbone_name=args.backbone_name,
        batch_size=args.batch_size,
        device=args.device,
        finetune=args.finetune,
        lr=args.lr,
        num_epochs=args.num_epochs,
        binary=args.binary,
    )