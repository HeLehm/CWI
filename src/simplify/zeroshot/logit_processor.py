"""
custom beam search based on cwi
NOTE: This is  not feasable 
"""
import torch
from transformers import LogitsProcessor

from ...cwi.model import ModelForTokenRegression

from tqdm import tqdm

class CWILogits(LogitsProcessor):
    """
    Logits processor based on the CWI model, i.e. it favors tokens that are less complex.
    """

    def __init__(
            self,
            cwi_model_path,
            tokenizer,
            device="cpu",
            scale = 1.0,
            batch_size=128,
            top_n=128,
            *args, **kwargs
    ):
        """
        :param cwi_model_path: The path to the CWI model to use.
        :param device: The device to use.
        """
        super().__init__(*args, **kwargs)
        self.model = ModelForTokenRegression.load(cwi_model_path, device=device)
        self.keys = list(tokenizer.vocab.keys())
        self.tokenizer = tokenizer
        self.scale = scale
        self.batch_size = batch_size
        self.top_n = top_n

    def __call__(self, input_ids, scores):
        """
        :param input_ids: The input ids, shape (num_beams, seq_len).
        :param scores: The scores, shape (num_beams, vocab length).
        :return: The scores multiplied by the CWI score.
        NOTE: vocab length example for T5: 32128
        """

        # create input idsl like this:
        # for every beam, for every token, create a new input  and attention mask
    
        # for every beam
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):

            # for every possible next token (self.keys)
            # in batches of 32

            # try only the top n
            top_tokens = torch.topk(beam_scores, self.top_n, dim=-1).indices

            # set other scores to -inf
            mask = torch.ones_like(beam_scores) * float('-inf')
            mask[top_tokens] = 0
            beam_scores += mask


            for batch_start in range(0, self.top_n, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.top_n)
                batch_keys = [self.keys[token_idx] for token_idx in top_tokens[batch_start:batch_end]]
                batch_input_ids = torch.cat([beam_input_ids.unsqueeze(0)] * len(batch_keys), dim=0)
                # add new token to the end of the input ids
                batch_input_ids = torch.cat([batch_input_ids, torch.zeros_like(batch_input_ids[:, -1]).unsqueeze(-1)], dim=1)
                batch_input_ids[:, -1] = torch.tensor([self.tokenizer.vocab[key] for key in batch_keys])
                batch_attention_mask = torch.ones_like(batch_input_ids)

                batch_input_ids = batch_input_ids.to(self.model.regression.weight.device)
                batch_attention_mask = batch_attention_mask.to(self.model.regression.weight.device)

                # get the logits
                logits = self.model(batch_input_ids, batch_attention_mask)
                # maske sure the logits are in range 0, 1
                logits = torch.clamp(logits, 0., 1.)
                # flip logits ( because low is good)
                logits = 1. - logits
                # average logits over the sequence
                logits = logits.mean(dim=1)
            
                logits = logits.squeeze(1)

                # log scale
                logits = torch.log(logits)

                # scale the logits
                logits = logits * self.scale
                
                # add the logits to the scores
                beam_scores[batch_start:batch_end] -= logits.to(beam_scores.device)

        return scores