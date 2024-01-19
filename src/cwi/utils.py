import numpy as np

def interpolate_color(label_prob):
    # Interpolate between white (255, 255, 255) and red (255, 0, 0)
    # turn < 0 to 0
    label_prob = max(label_prob, 0)
    r = 255
    g = int(255 * (1 - label_prob))
    b = int(255 * (1 - label_prob))
    return r, g, b

def decode_with_mapping(input_ids, tokenizer):
    """
    Decodes the input ids to text and attempts to reconstruct a mapping from the decoded text to the original text indices.
    
    Args:
    input_ids (list): List of token ids to be decoded.
    tokenizer: The tokenizer used for encoding.

    Returns:
    tuple: Decoded text and list of mappings (start, end) for each decoded word to the original text.
    """
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    mappings = []
    current_position = 0

    for token_id in input_ids:
        token = tokenizer.decode([token_id], skip_special_tokens=True)
        if token.strip():  # Ignore special and padding tokens
            start_position = decoded_text.find(token, current_position)
            end_position = start_position + len(token)
            current_position = end_position
            mappings.append((start_position, end_position))
        else:
            mappings.append((None, None))  # For special tokens, we return None mapping

    return decoded_text, mappings

def depict_sample(input_ids, logits, tokenizer, html=False):
    """
    depict a colored sample
    """
    input_ids = input_ids.cpu().detach().numpy()
    logits = logits.cpu().detach().numpy()
    
    text, mapping = decode_with_mapping(input_ids, tokenizer)

    # color in each token
    new_text = ""
    last_end_index = 0
    for i, (start, end) in enumerate(mapping):
        if start is None:
            continue
        r, g, b = interpolate_color(logits[i])
        # Add the uncolored text before the current token
        new_text += text[last_end_index:start]
        # Add the colored token
        if html:
            new_text += f"<span style='background-color:rgb({r},{g},{b})'>{text[start:end]}</span>"
        else:
            new_text += f"\033[38;2;{r};{g};{b}m{text[start:end]}\033[0m"
        last_end_index = end
            
    return new_text