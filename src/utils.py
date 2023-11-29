import numpy as np

def interpolate_color(label_prob):
    # Interpolate between white (255, 255, 255) and red (255, 0, 0)
    # turn < 0 to 0
    label_prob = max(label_prob, 0)
    r = 255
    g = int(255 * (1 - label_prob))
    b = int(255 * (1 - label_prob))
    return r, g, b

def depict_sample(input_ids, logits , tokenizer):
    """
    depict a colored sample
    """
    input_ids = input_ids.cpu().detach().numpy()
    logits = logits.cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # ignore padding
    tokens = tokens[:np.array(input_ids).nonzero()[0][-1]]
    
    result = ""
    for i,token in enumerate(tokens):
        color = interpolate_color(logits[i])
        result += '\033[38;2;{};{};{}m'.format(color[0], color[1], color[2])
        result += token.replace('##', '')

        end = ' ' if len(tokens) >= i + 2 and not tokens[i+1].startswith('##') else ''
        result += '\033[0m' + end
    
    return result

    



