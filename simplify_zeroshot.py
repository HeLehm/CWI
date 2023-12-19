import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.simplify.zeroshot.logit_processor import CWILogits

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
    max_length=128,
    
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
        max_length=128,
        cwi=True,
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(model.device)


    # instantiate logits processor
    processors = []
    if cwi:
        processors.append(CWILogits(
            cwi_model_path="./models/cwi/humarin/chatgpt_paraphraser_on_T5_base_adapter_0.001_10_False",
            tokenizer=tokenizer,
            device=device,
            scale=10.0,
            batch_size=128,
            top_n=num_beams * 2,
        ))
    logits_processor = LogitsProcessorList(processors)


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
    prompt = "In the realm of zoological taxonomy, the panthera leo, commonly known as the lion, exhibits a fascinating array of behavioral adaptations that enhance its predatory efficacy."
    print("regular:", *paraphrase_beam_search(prompt, cwi=False, num_return_sequences=1))
    print("simple:", *paraphrase_beam_search(prompt, cwi=True, num_return_sequences=1))