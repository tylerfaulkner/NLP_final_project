from mvp import compute_metrics, toDataset, process_data_to_model_inputs
import numpy as np
import pickle
from functools import partial
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import LEDTokenizer
from transformers import LEDForConditionalGeneration, LEDConfig

def train_model():
    """
    Training code from script-2-story repo https://github.com/tony-hong/script-2-story/blob/main/train.py
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/led-base-16384")
    # max encoder length for led
    encoder_max_length = 1024
    decoder_max_length = 768
    batch_size = 1
    gradient_accumulation_steps = 4
    noise_lambda = 0
    learning_rate = 5e-5
    weight_decay = 0.01
    num_train_epochs = 20

    num_samples = 1022
    num_steps = float(num_samples) * num_train_epochs / \
        (batch_size * gradient_accumulation_steps)
    steps_per_epoch = int(num_steps / num_train_epochs)
    # Load the longformer from huggingface
    led = AutoModelForSeq2SeqLM.from_pretrained(
        "allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)  # Load the tokenizer
    # tokenized_data = tokenizer(data, return_tensors='pt')
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        fp16=False,
        fp16_backend="apex",
        output_dir="./",
        logging_steps=steps_per_epoch,
        eval_steps=steps_per_epoch,
        save_steps=steps_per_epoch,
        warmup_steps=512,
        save_total_limit=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adafactor"
    )

    led.config.num_beams = 2
    led.config.max_length = 1024
    led.config.min_length = 768
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3
    rouge = load_metric("rouge")
    compute_metrics_partial = partial(
        compute_metrics, tokenizer=tokenizer, rouge=rouge)

    # Perform k-fold cross validation
    k = 6
    # Split data into k folds
    data = pickle.load("./scriptbase_formatting/scriptbase_alpha_list") # indexes should match
    training_split = 0.8
    train_set_array = data[:int(len(data) * training_split)]
    val_set_array = data[int(len(data) * training_split):]
    print("Rows in train array:" + str(len(train_set_array)))
    # Get validation data
    print("Rows in val array:" + str(len(val_set_array)))
    print(len(val_set_array[:, 0]))
    # TODO: CONVERT DATA TO DATASET
    train_set = toDataset(train_set_array)
    val_set = toDataset(val_set_array)
    def tokenize_function(examples):
        return tokenizer(examples["scripts"], padding="max_length", truncation=True)
    train_set = train_set.map(
        process_data_to_model_inputs, batched=True, batch_size=batch_size, remove_columns=["movies", "summaries", "scripts"])
    val_set = val_set.map(
        process_data_to_model_inputs, batched=True, batch_size=batch_size, remove_columns=["movies", "summaries", "scripts"])
    
    train_set.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    val_set.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    print("Datasets Converted to Input Format")
    print(train_set)
    print(val_set)
    # Train model on training data
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=led,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics_partial,
        train_dataset=train_set,
        eval_dataset=val_set,
    )

    # start training
    # torch.autograd.set_detect_anomaly(True)
    trainer.train()
    trainer.evaluate()
    trainer.save_model("scriptbase_model/")


if __name__ == "__main__":
    train_model()
