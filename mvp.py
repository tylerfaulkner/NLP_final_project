# importing libraries
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from torch.utils.data import Dataset
import numpy as np
# import evaluate
import os
import re

# List generated with the help of Github Copilot
SCRIPT_STOP_WORDS = ['EXT.', ' EXT ', '- NIGHT', '- DAY',
                     'INT.', 'IN:', 'BACK TO SCENE', "(CONT'D)", '- CONTINUOUS', ' CONTINUOUS ', ' INT ',
                     ' FADE ', 'FADE OUT', 'FADE IN', 'FADE TO BLACK', 'FADE TO:', 'FADE TO',
                     'CUT TO:', 'CUT TO', 'CUT TO BLACK', 'CUT TO BLACK:',
                     'DISSOLVE TO:', 'DISSOLVE TO', 'DISSOLVE TO BLACK:',
                     'CONTINUED:', 'CONTINUED', 'CONTINUED ON',
                     'CONTINUED ON:', 'CONTINUED FROM', 'CONTINUED FROM:', 'CONTINUED IN', 'CONTINUED IN:', 'CONTINUED']

# Input text - to summarize


def summarize_text(text, threshold=1):
    """
    Extractive summarizer from https://www.geeksforgeeks.org/python-text-summarizer/
    """
    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Get token count
    print("Total amount of token:", len(words))

    # Creating a frequency table to keep the
    # score of each word

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (threshold * average)):
            summary += " " + sentence
    summary_tokens = word_tokenize(summary)
    print("Total amount of token in summary:", len(summary_tokens))
    print("Summary:", summary)
    return summary, len(summary_tokens)


def reduceScriptTo16k(text):
    # Reduce script to 16k tokens
    threshold = 0.5
    summary, tokens, = summarize_text(text, threshold)
    while tokens > 16384:
        threshold += 0.1
        summary, tokens = summarize_text(text, threshold)
    return summary

# compute Rouge score during validation


def compute_metrics(pred, tokenizer, rouge):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str)
    avg_score = (rouge_output["rouge1"].mid.fmeasure+rouge_output["rouge2"].mid.fmeasure +
                 rouge_output["rougeL"].mid.fmeasure +
                 rouge_output["rougeLsum"].mid.fmeasure
                 )/4

    return {
        "rouge1_precision": round(rouge_output["rouge1"].mid.precision, 4),
        "rouge1_recall": round(rouge_output["rouge1"].mid.recall, 4),
        "rouge1_fmeasure": round(rouge_output["rouge1"].mid.fmeasure, 4),

        "rouge2_precision": round(rouge_output["rouge2"].mid.precision, 4),
        "rouge2_recall": round(rouge_output["rouge2"].mid.recall, 4),
        "rouge2_fmeasure": round(rouge_output["rouge2"].mid.fmeasure, 4),

        "rougeL_precision": round(rouge_output["rougeL"].mid.precision, 4),
        "rougeL_recall": round(rouge_output["rougeL"].mid.recall, 4),
        "rougeL_fmeasure": round(rouge_output["rougeL"].mid.fmeasure, 4),

        "rougeLsum_precision": round(rouge_output["rougeLsum"].mid.precision, 4),
        "rougeLsum_recall": round(rouge_output["rougeLsum"].mid.recall, 4),
        "rougeLsum_fmeasure": round(rouge_output["rougeLsum"].mid.fmeasure, 4),

        "average_rogue": round(avg_score, 4),
    }


def loadScriptData():
    # Load all scripts in the clean_scripts folder
    script_data = []
    for filename in os.listdir("clean_scripts"):
        with open(os.path.join("clean_scripts", filename), "r") as f:
            script_data.append((removeScriptWords(f.read()),
                               filename.replace(".txt", "")))
    # Load all summaries inf the summaries folder
    summary_data = []
    for filename in os.listdir("summaries"):
        with open(os.path.join("summaries", filename), "r") as f:
            summary_data.append((f.read(), filename.replace(".txt", "")))
    # Combine the data by matching the filenames
    data = []
    for script in script_data:
        for summary in summary_data:
            if script[1] == convertFileName(summary[1]):
                data.append([script[0], summary[0]])
    return data  # [:,0], data[:,1]


def convertFileName(filename):
    # Convert spaces into dashes
    temp = filename.replace(":", "")
    temp = re.sub(r'\([^)]*\)', '', temp)
    temp = temp.strip()
    temp = temp.replace(" ", "-")
    # Remove all parentheses and characters between
    # Move The to the end of the filename
    if "The" in temp:
        temp = temp.replace("The-", "")
        temp = temp + ",-The"
    return temp


class MovieDataset(Dataset):
    # Pytorch Dataset
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx][0], self.data_list[idx][1]


def train_model():
    """
    Training code from script-2-story repo https://github.com/tony-hong/script-2-story/blob/main/train.py
    """
    # max encoder length for led
    encoder_max_length = 1024
    decoder_max_length = 768
    batch_size = 16
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
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/led-base-16384")  # Tokenize the data
    tokenized_data = tokenizer(data, return_tensors='pt')
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        fp16=False,
        #     fp16_backend="apex",
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

    # TODO: load all scripts and summaries into a dataset
    data = loadScriptData()  # indexes should match

    # Perform k-fold cross validation
    k = 6
    # Split data into k folds
    folds = np.array_split(data, k)
    # For each fold, train on k-1 folds and validate on the remaining fold
    for i in range(k):
        # Get training data
        train_set_array = folds[:i] + folds[i+1:]
        # Get validation data
        val_set_array = folds[i]
        # TODO: CONVERT DATA TO DATASET
        train_set = MovieDataset(train_set_array)
        val_set = MovieDataset(val_set_array)
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


def removeScriptWords(text):
    """
    Remove words that are commonly found in scripts
    """
    # Remove script stop words
    for word in SCRIPT_STOP_WORDS:
        text = text.replace(word, '')
    return text


if __name__ == "__main__":
    # Read clean script
    with open('clean_scripts/John-Wick.txt', 'r') as file:
        data = file.read()
        # Split data into scenes
        # Each scene begins with a line that starts with a number
        scenes = data.split('\n')
        # Remove empty lines
        scenes = [line for line in scenes if line.strip() != '']
        # Find the indexes of all lines that start with a number
        scene_indexes = [i for i, line in enumerate(
            scenes) if line[0].isdigit()]
        # Split the data into scenes
        scenes = [scenes[i:j]
                  for i, j in zip(scene_indexes, scene_indexes[1:]+[None])]
        print(len(scenes))
        # print(scenes[0])
        # combine each scene into one string
        scenes = [' '.join(scene) for scene in scenes]
        # print(convertFileName("The Bourne Identity (2002 film)"))
        summarize_text(removeScriptWords(data))
