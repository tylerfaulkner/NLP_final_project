import pandas as pd
import torch
import pandas as pd
from datasets import Dataset
from datasets import load_dataset, load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import logging
import time

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# load rouge
rouge = load_metric("rouge")
print("We recommend to use a GPU to speed up inference time, but this can be ran on a CPU. It will take much longer though.")
movie_path = input("Enter the path to the movie script (example, test_data/Alien_script.txt): ")

# load testset
script, summ = "", ""
with open(movie_path, "r") as f:
    script = f.read()
sum_path = input("Enter the path to the movie summary (example, test_data/Alien_summ.txt): ")
with open(sum_path, 'r') as f:
    summ = f.read()

test_set = Dataset.from_dict({"script": [script], "summary": [summ]})

# load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("grizzlypath26/script2sumPrototype")
# led = AutoModelForSeq2SeqLM.from_pretrained(
#     "allenai/led-large-16384", gradient_checkpointing=False, use_cache=False)
#tokenizer = AutoTokenizer.from_pretrained(
 #   "check")
print("Loading model...")
led = AutoModelForSeq2SeqLM.from_pretrained(
    "grizzlypath26/script2sumPrototype", use_cache=False)

#led = BetterTransformer.transform(led, keep_original_model=True)

# load tokenizer
#TODO swithc back to cuda 
model = led.to(device)


def generate_answer(batch):
    print("Tokenizing batch...")
    inputs_dict = tokenizer(batch["script"], padding="max_length", max_length=1024, return_tensors="pt", truncation=True)
    # input_ids = inputs_dict.input_ids.to("cuda")
    # attention_mask = inputs_dict.attention_mask.to("cuda")

    input_ids = inputs_dict.input_ids.to(device)
    attention_mask = inputs_dict.attention_mask.to(device)
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    print("Generating summary from model...")
    #get time for inference
    start = time.time()
    predicted_summary_ids = model.generate(
        input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    print("Decoding summary...")
    batch["predicted_summary"] = tokenizer.batch_decode(
        predicted_summary_ids, skip_special_tokens=True)
    end = time.time()
    print("Time for inference (s): ", end - start)
    return batch

print("Generating summary...")
result = test_set.map(generate_answer, batched=True, batch_size=1)
print(result["predicted_summary"])
print("Result (Rouge Score):", rouge.compute(predictions=result["predicted_summary"],
      references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid)

with open("output_score.txt", "a") as f:
    print("Result:", rouge.compute(predictions=result["predicted_summary"],
          references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid, file=f)

pd.DataFrame(result).to_csv("result.csv")
