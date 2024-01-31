import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

from src.cwi.data import ComplexWordDataset
from src.cwi.model import predict_batch, ModelForTokenRegression
from src.cwi.utils import depict_sample
from src.cwi.data.utils import COLUMNS


torch.set_num_threads(1)


def load_test_set(path):
    """
    loads the data as a pandas dataframe
    """
    # load data
    path = "/Users/hergen/Desktop/UHH/MSc/FS 1/NLP/CWI/Assignment-Option-3-cwi-datta/option-3-test/News_Test.tsv"

    df = pd.read_csv(path, sep='\t', names=COLUMNS)

    return df

def main(model_path, device="cpu", data_path=None):

    model = ModelForTokenRegression.load(model_path, device=device)
    binary = model_path.split("/")[-1].split("_")[-1] == "True"
    tokenizer = AutoTokenizer.from_pretrained(model.backbone_name)
    model = model.to(device)
    model.eval()

    # Loss function
    if binary:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.MSELoss()

    d = load_test_set(data_path)
    dev_ds = ComplexWordDataset(d, tokenizer)
    dev_dl = dev_ds.get_dataloader(batch_size=1, shuffle=False)

    final_text = ""

    with torch.no_grad():
        total_val_loss = 0
        total_val_seen = 0
        
        for batch in tqdm(dev_dl):
            outputs, label_probs, token_ids = predict_batch(model, batch, device, binary=binary)
            loss = criterion(outputs, label_probs)
            total_val_loss += loss.item()
            total_val_seen += len(batch)
            
            final_text += "Predicted:\n"
            final_text += depict_sample(token_ids[0], outputs[0], tokenizer)
            final_text += "\n"
            final_text += "Label:\n"
            final_text += depict_sample(token_ids[0], label_probs[0], tokenizer)
            final_text += "\n\n"

    # save final text
    with open("./result_samples.txt", "w") as f:
        f.write(final_text)

    print(f"Average Validation Loss = {total_val_loss / total_val_seen}")
    print("Saved result_samples.txt")
    print(final_text)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to model directory", default="./models/cwi/bert-base-uncased_adapter_0.001_10_False")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--data_path", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.model, device=args.device, data_path=args.data_path)

