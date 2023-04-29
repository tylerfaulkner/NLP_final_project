import pandas as pd
import numpy as np
import torch
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset
from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# load rouge
rouge = load_metric("rouge")

# load testset
script, summ = "", ""
with open("test_data/Alien_script.txt", "r") as f:
    script = f.read()
with open('test_data/Alien_summ.txt', 'r') as f:
    summ = f.read()

test_set = Dataset.from_dict({"script": [script], "summary": [summ]})

# load tokenizer

# tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")
# led = AutoModelForSeq2SeqLM.from_pretrained(
#     "allenai/led-large-16384", gradient_checkpointing=False, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(
    "pre_trained_model")
led = AutoModelForSeq2SeqLM.from_pretrained(
    "pre_trained_model", use_cache=False)

# load tokenizer
#TODO swithc back to cuda 
# model = led.to("cuda").half()
model = led.to("cpu")


def generate_answer(batch):
    inputs_dict = tokenizer(batch["script"], padding="max_length",
                            max_length=8192, return_tensors="pt", truncation=True)
    # input_ids = inputs_dict.input_ids.to("cuda")
    # attention_mask = inputs_dict.attention_mask.to("cuda")
    input_ids = inputs_dict.input_ids.to("cpu")
    attention_mask = inputs_dict.attention_mask.to("cpu")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_summary_ids = model.generate(
        input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    batch["predicted_summary"] = tokenizer.batch_decode(
        predicted_summary_ids, skip_special_tokens=True)
    return batch


result = test_set.map(generate_answer, batched=True, batch_size=1)

print("Result:", rouge.compute(predictions=result["predicted_summary"],
      references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid)

with open("output_score.txt", "a") as f:
    print("Result:", rouge.compute(predictions=result["predicted_summary"],
          references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid, file=f)

pd.DataFrame(result).to_csv("result.csv")
