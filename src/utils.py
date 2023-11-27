import numpy as np

def interpolate_color(label_prob):
    # Interpolate between white (255, 255, 255) and red (255, 0, 0)
    r = 255
    g = int(255 * (1 - label_prob))
    b = int(255 * (1 - label_prob))
    return r, g, b

def depict_sample(input_ids, logits , tokenizer):
    """
    depict a colored sample
    """
    input_ids = input_ids.cpu().numpy()
    logits = logits.cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # ignore padding
    tokens = tokens[:np.array(input_ids).nonzero()[0][-1]]
    
    for i,token in enumerate(tokens):
        color = interpolate_color(logits[i])
        print('\033[38;2;{};{};{}m'.format(color[0], color[1], color[2]), end='')
        print(token.replace('##', ''), end='')
        end = ' ' if len(tokens) >= i + 2 and not tokens[i+1].startswith('##') else ''
        print('\033[0m', end=end)
    print()

    



