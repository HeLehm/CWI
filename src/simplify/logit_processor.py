"""
custom beam search based on cwi
NOTE: This is  not feasable 
"""
import torch
from transformers import LogitsProcessor

from ..cwi.model import ModelForTokenRegression

from tqdm import tqdm

class CWILogits(LogitsProcessor):
    """
    Logits processor based on the CWI model, i.e. it favors tokens that are less complex.
    """

    def __init__(self, cwi_model_path, tokenizer, device="cpu", scale = 1.0, batch_size=128, *args, **kwargs):
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

    def __call__(self, input_ids, scores):
        """
        :param input_ids: The input ids, shape (num_beams, seq_len).
        :param scores: The scores, shape (num_beams, vocab length).
        :return: The scores multiplied by the CWI score.
        NOTE: vocab length example for T5: 32128
        """
        print("CWI processor called")
        print("input_ids", input_ids.shape)
        print("scores", scores.shape)

        # create input idsl like this:
        # for every beam, for every token, create a new input  and attention mask
    
        # for every beam
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):

            # for every possible next token (self.keys)
            # in batches of 32

            for batch_start in tqdm(range(0, len(self.keys), self.batch_size), desc=f"CWI {beam_index}/{len(input_ids)}"):
                batch_end = batch_start + self.batch_size
                batch_keys = self.keys[batch_start:batch_end]
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
                # flip logits ( becasue low is good)
                logits = 1. - logits
                # average logits over the sequence
                logits = logits.mean(dim=1)
                # scale the logits
                logits = logits * self.scale

                logits = logits.squeeze(1)
                
                # add the logits to the scores
                beam_scores[batch_start:batch_end] += logits.to(beam_scores.device)
            
            

    
"""
EXAMPLE:
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.simplify.logit_processor import CWILogits

from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
    MaxLengthCriteria
)

device = "mps"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)


def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(model.device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


def paraphrase_beam_search(
        question,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(model.device)


    # instantiate logits processor
    logits_processor = LogitsProcessorList([
        CWILogits("./models/cwi/humarin/chatgpt_paraphraser_on_T5_base_adapter_0.001_10_False", device="cpu", tokenizer=tokenizer, scale=1.0),
    ])

    input_ids=torch.cat([input_ids] * num_beam_groups, dim=0)



    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

if __name__ == '__main__':
    prompt = "What are some must see places in New York City?"
    print(paraphrase_beam_search(prompt))
"""