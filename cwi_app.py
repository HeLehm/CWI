from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer
import os
import numpy as np

app = Flask(__name__)

model = None
tokenizer = None
device = None


def get_model_path():
    # look into ./models folder
    models_dir = os.path.join(os.getcwd(), 'models', 'cwi')
    # get first models
    model_path = os.path.join(models_dir, os.listdir(models_dir)[0])
    return model_path


def load_model(model_path, device_="cpu"):
    global model, tokenizer, device
    device = device_
    # Load model and tokenizer
    model = torch.load(model_path)
    backbone_name = model_path.split("/")[-1].split("_")[0]
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    model = model.to(device)
    model.eval()


def interpolate_color(value):
    # This function should return an RGB color based on the value.
    # Implement your own logic here to convert a value to a color.
    # For example, a simple linear interpolation:
    red = int(255 * value)
    green = int(255 * (1 - value))
    blue = 0  # Keeping blue constant
    return red, green, blue



def depict_sample_html(input_ids, logits, tokenizer):
    """
    Depict a colored sample for HTML output.
    """
    input_ids = input_ids.cpu().detach().numpy()
    logits = logits.cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    tokens = tokens[:np.array(input_ids).nonzero()[0][-1]]  # ignore padding
    # ignore cls token
    tokens = tokens[1:]
    logits = logits[1:]
    
    result = ""
    for i, token in enumerate(tokens):
        color = interpolate_color(logits[i])
        colored_token = f'<span style="color: rgb({color[0]}, {color[1]}, {color[2]})">{token.replace("##", "")}</span>'
        end = ' ' if len(tokens) >= i + 2 and not tokens[i+1].startswith('##') else ''
        result += colored_token + end
    
    return result


# Define the route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input text from the form
        input_text = request.form['input_text']
        processed_text = process_text(input_text)
        return render_template('index.html', processed_text=processed_text)
    return render_template('index.html')

def process_text(input_text):
    encoding = tokenizer(
            input_text,
            return_offsets_mapping=True,
            max_length=512,
            padding='max_length',
        )
    input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).unsqueeze(0)
    outputs = model(
        input_ids.to(device),
        attention_mask=attention_mask.to(device)
    )
    processed_text = depict_sample_html(input_ids[0], outputs[0], tokenizer)
    return processed_text
    

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    if model_path is None:
        model_path = get_model_path()
    load_model(model_path, device_=args.device)

    app.run(debug=True)
