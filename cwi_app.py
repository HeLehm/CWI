from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer
from src.cwi.model import ModelForTokenRegression
from src.cwi.utils import depict_sample
import os
import numpy as np

app = Flask(__name__)

model = None
tokenizer = None
device = None


def load_model(model_path, device_="cpu"):
    global model, tokenizer, device
    device = device_
    model = ModelForTokenRegression.load(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model.backbone_name)


def interpolate_color(value):
    # This function should return an RGB color based on the value.
    # Implement your own logic here to convert a value to a color.
    # For example, a simple linear interpolation:
    red = int(255 * value)
    green = int(255 * (1 - value))
    blue = 0  # Keeping blue constant
    return red, green, blue


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
    processed_text = depict_sample(input_ids[0], outputs[0], tokenizer, html=True)
    return processed_text
    

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/cwi/humarin/chatgpt_paraphraser_on_T5_base_adapter_0.001_10_False")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    load_model(model_path, device_=args.device)

    app.run(debug=True)
