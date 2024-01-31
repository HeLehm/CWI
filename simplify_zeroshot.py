import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.simplify.zeroshot import CWRLogitsProcessor
from src.simplify.visualization import TwoStepBeamSearchDataLogger
from src.cwi.utils import depict_sample

from tqdm import tqdm

from transformers import LogitsProcessorList

import json



def paraphrase_beam_search(
        question,
        model,
        tokenizer,
        device,
        cwi_model_path="./models/cwi/humarin/chatgpt_paraphraser_on_T5_base_adapter_0.001_10_False",
        num_beams=3,
        num_return_sequences=None,
        no_repeat_ngram_size=2,
        max_length=64,
        cwi_top_n=64,
        vis_data_output_path=None,
):
    
    if num_return_sequences is None:
        num_return_sequences = num_beams

    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(model.device)

    prog_bar = None
    def linact(x):
        x = 20 * x
        x = torch.clamp(x,-1.,1.)
        return x

    prog_bar = tqdm(range(max_length * num_beams), desc="Paraphrasing")
    cwi_p = CWRLogitsProcessor(
            cwi_model_path=cwi_model_path,
            tokenizer=tokenizer,
            device=device,
            top_n=cwi_top_n,
            prog_bar=prog_bar,
            loss_activation=linact,
        )
    cwi_p.to(device)

    beam_logger = TwoStepBeamSearchDataLogger(
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        top_k=cwi_top_n,
    )


    logits_processor = LogitsProcessorList(
        [
            beam_logger,
            cwi_p,
            beam_logger,
        ]
    )

    # set seed
    torch.manual_seed(42)

    with torch.no_grad():
        # uses regular beam search
        outputs = model.generate(
            input_ids,
            num_beams=num_beams,

            max_new_tokens=max_length,
            num_return_sequences=num_beams,
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

    # calculate complexity loss
    loss = losses.sum(-1).tolist()

    # store beam search data
    if vis_data_output_path is not None:
        data = beam_logger.get_data()
        with open(vis_data_output_path, "w") as f:
            json.dump(data, f)


    return res, loss

if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--cwi_path", type=str, default="./models/cwi/humarin/chatgpt_paraphraser_on_T5_base_adapter_0.001_10_False")
    parser.add_argument("--cwi_top_n", type=int, default=64)
    parser.add_argument("--paraphraser_path", type=str, default="humarin/chatgpt_paraphraser_on_T5_base")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--vis_data_output_path", type=str, default=None)

    args = parser.parse_args()


    model = AutoModelForSeq2SeqLM.from_pretrained(args.paraphraser_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.paraphraser_path)


    avg_s_loss = 0
    s_sentence, s_loss = paraphrase_beam_search(
        args.prompt,
        model,
        tokenizer,
        args.device,
        cwi_model_path=args.cwi_path,
        num_beams=args.num_beams,
        cwi_top_n=args.cwi_top_n,
        vis_data_output_path=args.vis_data_output_path,
    )
    print("Results:")
    for i, (sentence, loss) in enumerate(zip(s_sentence, s_loss)):
        print(f"{sentence} ({loss})")
        avg_s_loss += loss
    avg_s_loss /= len(s_loss)
    print()
    print(f"average simple loss: {avg_s_loss}")
    print(f"lowest simple loss: {min(s_loss)}")
    print()