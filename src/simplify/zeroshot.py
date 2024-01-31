"""
custom beam search based on cwi
NOTE: This is  not feasable 
"""
import torch
from transformers import LogitsProcessor

from ..cwi.model import ModelForTokenRegression

from typing import Optional, Callable

class CWRLogitsProcessor(LogitsProcessor):
    """
    Logits processor based on the CWI model, i.e. it favors tokens that are less complex.
    """

    def __init__(
            self,
            cwi_model_path,
            tokenizer,
            device="cpu",
            loss_activation : Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
            top_n=128,
            prog_bar=None,
            *args, **kwargs
    ):
        """
        :param cwi_model_path: The path to the CWI model to use.
        :param device: The device to use.
        """
        super().__init__(*args, **kwargs)

        self.model = ModelForTokenRegression.load(cwi_model_path, device=device)
        self.tokenizer = tokenizer
        self.top_n = top_n
        self.prog_bar = prog_bar
        self.loss_activation = loss_activation

    def __call__(self, input_ids, scores):
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
            scores[element_idx, :] += mask

            # cwi inputs (present input ids + new (top) token)
            # for every token, create a new input  and attention mask
            # +1 beause last is just the sum loss right now
            cwi_input_ids = torch.cat([element_input_ids.unsqueeze(0)] * (len(top_tokens_indices) + 1), dim=0)
            # add dimension for the new token
            cwi_input_ids = torch.cat([cwi_input_ids, torch.zeros_like(cwi_input_ids[:, -1]).unsqueeze(-1)], dim=1)
            # add thetop tokens
            cwi_input_ids[:-1, -1] = top_tokens_indices.detach().to(cwi_input_ids.device)
            # set to pad token or base loss
            cwi_input_ids[-1, -1] = self.tokenizer.pad_token_id

            cwi_input_ids = cwi_input_ids.to(self.model.regression.weight.device)
            # cwi pass
            # wil be number from 0 to 1
            # 0 = simple -> 1 = complex
            losses = self.loss(cwi_input_ids, mode="sum")

            base_loss = losses[-1]
            losses = losses[:-1]
            
            # loss is based on change in sum
            losses = losses - base_loss

            # apply activation function to loss
            losses = self.loss_activation(losses)

            losses = torch.log_softmax(losses, dim=-1)

            scores[element_idx, top_tokens_indices] = torch.log_softmax(scores[element_idx, top_tokens_indices], dim=-1)

            scores[element_idx, top_tokens_indices] -= losses



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


    def loss(self, input_ids, mode: Optional[str] = None):
        """
        caluclates loss input_ids based on the cwi model
        
        :param input_ids: shape (batch_size, seq_len)

        returns shape (batch_size)
        """
        logits = self.cwi(input_ids)
        
        if mode is None:
            mode = ""
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
    
    def to(self, device):
        self.model = self.model.to(device)