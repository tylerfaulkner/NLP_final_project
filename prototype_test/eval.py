import torch
import pandas as pd
from datasets import Dataset
from datasets import load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import time

def get_tensor_rank(tensor):
    #check tensor dtype
    if tensor.dtype != torch.float32:
        tensor = tensor(0)
    return torch.linalg.matrix_rank(tensor)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_index = 0
print("Device: ", device)
print("Device index: ", device_index)

default_script = "test_data/Alien_script.txt"
default_summ = "test_data/Alien_summ.txt"

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
# load tokenizer
#TODO swithc back to cuda 
model = led.to(device)
if torch.cuda.is_available():
    model = model.half()
else:
    #optimize for intel
    print("Optimizing for Intel...")
    import intel_extension_for_pytorch as ipex
    model = ipex.optimize(model, dtype=torch.bfloat16)
    model = model.to(memory_format=torch.channels_last)
model.eval()
with torch.no_grad(), torch.cpu.amp.autocast():
    # load rouge
    rouge = load_metric("rouge")
    print("We recommend to use a GPU to speed up inference time, but this can be ran on a CPU. It will take much longer though.")
    movie_path = input("Enter the path to the movie script (example, test_data/Alien_script.txt): ")
    if movie_path == "":
        movie_path = default_script
    
    movie_path = "tempScript.txt"

    # load testset
    script, summ = "", ""
    with open(movie_path, "r") as f:
        script = f.read()

    summ = ""

    test_set = Dataset.from_dict({"script": [script], "summary": [summ]})


    def generate_answer(batch):
        print("Tokenizing batch...")
        inputs_dict = tokenizer(batch["script"], padding="max_length", return_tensors="pt", truncation=True)
        # input_ids = inputs_dict.input_ids.to("cuda")
        # attention_mask = inputs_dict.attention_mask.to("cuda")

        input_ids = inputs_dict.input_ids.to(device)
        attention_mask = inputs_dict.attention_mask.to(device)
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1

        #if torch.cuda.device_count() < 1:
            #if get_tensor_rank(input_ids) > 3:
                #input_ids = input_ids.to(memory_format=torch.channels_last)
            #if get_tensor_rank(attention_mask) > 3:
             #   attention_mask = attention_mask.to(memory_format=torch.channels_last)
            #if get_tensor_rank(global_attention_mask) > 3:
             #   global_attention_mask = global_attention_mask.to(memory_format=torch.channels_last)
        # put global attention on <s> token
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
    with open("tempSumm.txt", "w") as f:
        f.write(result["predicted_summary"])
    #print("Result (Rouge Score):", rouge.compute(predictions=result["predicted_summary"],
       # references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid)

    #with open("output_score.txt", "a") as f:
     #   print("Result:", rouge.compute(predictions=result["predicted_summary"],
      #      references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid, file=f)

    #pd.DataFrame(result).to_csv("result.csv")