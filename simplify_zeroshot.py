import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.simplify.zeroshot.logit_processor import CWILogits

from tqdm import tqdm

from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
    MaxLengthCriteria
)

device = "mps"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)


def paraphrase_beam_search(
        question,
        num_beams=8,
        num_beam_groups=None,
        num_return_sequences=None,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=64,
        cwi=True,
        cwi_top_n=128,
):
    
    if num_beam_groups is None:
        num_beam_groups = num_beams
    
    if num_return_sequences is None:
        num_return_sequences = num_beams

    cwi_top_n = max(cwi_top_n, num_beams * 8)

    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(model.device)



    # instantiate logits processor
    processors = []
    prog_bar = None
    if cwi:
        prog_bar = tqdm(range(max_length * num_beams), desc="Paraphrasing")
    cwi_p = CWILogits(
            cwi_model_path="./models/cwi/humarin/chatgpt_paraphraser_on_T5_base_adapter_0.001_10_False",
            tokenizer=tokenizer,
            device=device,
            scale=100.,
            top_n=32,
            softmax=True,
            prog_bar=prog_bar,
        )
    if cwi:
        processors.append(cwi_p)
    logits_processor = LogitsProcessorList(processors)


    with torch.no_grad():
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

    # calculate complexity loss
    loss = cwi_p.loss(outputs, mode="sum").tolist()

    return res, loss

if __name__ == '__main__':
    #prompt = "In the realm of zoological taxonomy, the panthera leo, commonly known as the lion, exhibits a fascinating array of behavioral adaptations that enhance its predatory efficacy."

    prompt = "The proliferation of technologically advanced gadgets has substantially augmented the efficacy of our daily communications." # simple: The spread of high-tech devices has greatly improved how well we communicate every day

    avg_r_loss = 0
    r_sentence, r_loss = paraphrase_beam_search(prompt, cwi=False)
    for i, (sentence, loss) in enumerate(zip(r_sentence, r_loss)):
        print(f"regular: {sentence} ({loss})")
        avg_r_loss += loss
    avg_r_loss /= len(r_loss)
    print(f"average regular loss: {avg_r_loss}")

    avg_s_loss = 0
    s_sentence, s_loss = paraphrase_beam_search(prompt, cwi=True)
    for i, (sentence, loss) in enumerate(zip(s_sentence, s_loss)):
        print(f"simple: {sentence} ({loss})")
        avg_s_loss += loss
    avg_s_loss /= len(s_loss)
    print(f"average simple loss: {avg_s_loss}")