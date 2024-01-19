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
            scale=1.0,
            top_n=128,
            softmax=True,
            prog_bar=None,
            *args, **kwargs
    ):
        """
        :param cwi_model_path: The path to the CWI model to use.
        :param device: The device to use.
        """
        super().__init__(*args, **kwargs)

        assert scale > 0, "scale must be > 0"

        self.model = ModelForTokenRegression.load(cwi_model_path, device=device)
        self.tokenizer = tokenizer
        self.scale = scale
        self.top_n = top_n
        self.prog_bar = prog_bar
        self.softmax = softmax

        self.beam_search_data = []

    def __call__(self, input_ids, scores):
        """
        
        input_ids (torch.LongTensor of shape (batch_size, sequence_length))
            — Indices of input sequence tokens in the vocabulary.

        scores (torch.FloatTensor of shape (batch_size, config.vocab_size))
            — Prediction scores of a language modeling head.
            These can be logits for each vocabulary when not using beam search
            or log softmax for each vocabulary token when using beam search

        :return: The scores multiplied by the CWI score.
        NOTE: vocab length example for T5: 32128
        """

        # create input idsl like this:
        # for every beam, for every token, create a new input  and attention mask
        
        # for every item in batch 
        for element_idx, (element_input_ids, element_scores) in enumerate(zip(input_ids, scores)):

            # for every possible next token
            # try only the top n
            top_tokens_indices = torch.topk(element_scores, self.top_n, dim=-1).indices

            # if EOS token is top 1, pass
            if top_tokens_indices[0] == self.tokenizer.eos_token_id:
                continue

            # also continue if EOS is already in the beam
            if self.tokenizer.eos_token_id in element_input_ids:
                continue

            # set other scores to -inf
            mask = torch.ones_like(element_scores) * float('-inf')
            mask[top_tokens_indices] = 0
            # also dont mask EOS token
            # mask[self.tokenizer.eos_token_id] = 0
            scores[element_idx, :] += mask

            # cwi inputs (present input ids + new (top) token)
            # for every token, create a new input  and attention mask
            cwi_input_ids = torch.cat([element_input_ids.unsqueeze(0)] * len(top_tokens_indices), dim=0)
            # add dimension for the new token
            cwi_input_ids = torch.cat([cwi_input_ids, torch.zeros_like(cwi_input_ids[:, -1]).unsqueeze(-1)], dim=1)
            # add thetop tokens
            cwi_input_ids[:, -1] = top_tokens_indices.detach().to(cwi_input_ids.device)
            cwi_input_ids = cwi_input_ids.to(self.model.regression.weight.device)

            #cwi_input_ids_text = [self.tokenizer.decode(input_ids) for input_ids in cwi_input_ids]

            # cwi pass
            losses = self.loss(cwi_input_ids, mode="last")#.loss_decreasing(cwi_input_ids, max_window_size=6)

            # softmax
            if self.softmax:
                losses = torch.softmax(losses, dim=-1)
            
            # penalize scores based on the loss
            scores[element_idx, top_tokens_indices] *= torch.exp(-losses.to(element_scores.device) * self.scale)
            


        if self.prog_bar is not None:
            self.prog_bar.update(1)

        return scores
    
    def cwi(self, input_ids):
        """
        scores input_ids based on the cwi model
        high = simple
        low = complex

        :param input_ids: shape (batch_size, seq_len)

        returns shape (batch_size, seq_len)
        """
        input_ids = input_ids.to(self.model.regression.weight.device)
        # get the logits
        logits = self.model(input_ids)
        # maske sure the logits are in range 0, 1
        logits = torch.clamp(logits, 0., 1.)

        return logits.squeeze(-1)
    

    def loss_decreasing(self, input_ids, max_window_size=None):
        """
        input ids based on the cwi model, where the lates has the most wight and the first the least
        """
        logits = self.cwi(input_ids)

        # linearly scale the logits based on the position
        # where the last token has the most weight and the first the least
        # and only the last max_window_size tokens are considered
        if max_window_size is not None:
            logits = logits[:, -max_window_size:]
        logits = torch.linspace(0, 1, logits.shape[-1]).to(logits.device) * logits
        
        # take mean over the sequence
        logits = logits.mean(dim=-1)

        return logits
    
    def loss(self, input_ids, mode: str = "mean"):
        """
        caluclates loss input_ids based on the cwi model
        
        :param input_ids: shape (batch_size, seq_len)

        returns shape (batch_size)
        """
        logits = self.cwi(input_ids)
        
        # TODO: mean might not be the best messure, as the model can just extend the sentence
        # aggregate logits over the sequence
        if mode == "mean":
            logits = logits.mean(dim=-1)
        elif mode == "sum":
            logits = logits.sum(dim=-1)
        elif mode == "last":
            logits = logits[:, -1].squeeze(-1)
        elif mode.startswith("window"):
            window_size = int(mode.split("_")[1])
            logits = logits[:, -window_size:].mean(dim=-1)
        
        return logits

