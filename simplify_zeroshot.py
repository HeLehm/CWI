import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.simplify.zeroshot.logit_processor import CWILogits
from src.cwi.utils import depict_sample

from tqdm import tqdm

from transformers import LogitsProcessorList

device = "mps"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)





def paraphrase_beam_search(
        question,
        num_beams=8,
        num_return_sequences=None,
        no_repeat_ngram_size=2,
        max_length=64,
        cwi=True,
        cwi_top_n=64,
):
    
    if num_return_sequences is None:
        num_return_sequences = num_beams

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
        prog_bar = tqdm(range(64 * num_beams), desc="Paraphrasing")
    cwi_p = CWILogits(
            cwi_model_path="./models/cwi/humarin/chatgpt_paraphraser_on_T5_base_adapter_0.001_10_False",
            tokenizer=tokenizer,
            device=device,
            pow=20.0,
            top_n=cwi_top_n, # TODO: this should also be like the other top_n or top_p
            prog_bar=prog_bar,
        )
    if cwi:
        processors.append(cwi_p)
    logits_processor = LogitsProcessorList(processors)


    # set seed
    torch.manual_seed(42)

    with torch.no_grad():
        # uses regular beam search
        outputs = model.generate(
            input_ids,
            max_new_tokens=40,
            do_sample=True,
            top_p=0.92,
            #top_k=cwi_top_n,

            num_return_sequences=10,
            no_repeat_ngram_size=no_repeat_ngram_size,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    losses = cwi_p.loss(outputs.sequences, mode=None)
    res = []
    for i, seq in enumerate(outputs.sequences):
        res.append(
            depict_sample(
                seq,
                losses[i],
                tokenizer=tokenizer,
            )
        )
    #res = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

    # calculate complexity loss
    loss = losses.sum(-1).tolist()

    return res, loss

if __name__ == '__main__':
    #prompt = "In the realm of zoological taxonomy, the panthera leo, commonly known as the lion, exhibits a fascinating array of behavioral adaptations that enhance its predatory efficacy."

    prompt = "The proliferation of technologically advanced gadgets has substantially augmented the efficacy of our daily communications."
    # simple: The spread of high-tech devices has greatly improved how well we communicate every day

    avg_r_loss = 0
    r_sentence, r_loss = paraphrase_beam_search(prompt, cwi=False)
    for i, (sentence, loss) in enumerate(zip(r_sentence, r_loss)):
        print(f"regular: {sentence} ({loss})")
        avg_r_loss += loss
    avg_r_loss /= len(r_loss)
    print()
    print(f"average regular loss: {avg_r_loss}")
    print(f"lowest regular loss: {min(r_loss)}")
    print()

    avg_s_loss = 0
    s_sentence, s_loss = paraphrase_beam_search(prompt, cwi=True)
    for i, (sentence, loss) in enumerate(zip(s_sentence, s_loss)):
        print(f"simple: {sentence} ({loss})")
        avg_s_loss += loss
    avg_s_loss /= len(s_loss)
    print()
    print(f"average simple loss: {avg_s_loss}")
    print(f"lowest simple loss: {min(s_loss)}")
    print()