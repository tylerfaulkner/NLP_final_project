# importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import numpy as np
import evaluate

THRESHOLD = 1.5
#List generated with the help of Github Copilot
SCRIPT_STOP_WORDS = ['EXT.', ' EXT ', '- NIGHT', '- DAY',
                      'INT.', 'IN:','BACK TO SCENE', "(CONT'D)", '- CONTINUOUS', ' CONTINUOUS ', ' INT ',
                      ' FADE ', 'FADE OUT', 'FADE IN', 'FADE TO BLACK', 'FADE TO:', 'FADE TO',
                      'CUT TO:', 'CUT TO', 'CUT TO BLACK', 'CUT TO BLACK:',
                        'DISSOLVE TO:', 'DISSOLVE TO', 'DISSOLVE TO BLACK:',
                          'CONTINUED:', 'CONTINUED', 'CONTINUED ON',
                        'CONTINUED ON:', 'CONTINUED FROM', 'CONTINUED FROM:', 'CONTINUED IN', 'CONTINUED IN:', 'CONTINUED']
   
# Input text - to summarize 
def summarize_text(text):
    """
    Extractive summarizer from https://www.geeksforgeeks.org/python-text-summarizer/
    """
    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    #Get token count
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
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (THRESHOLD * average)):
            summary += " " + sentence
    summary_tokens = word_tokenize(summary)
    print("Total amount of token in summary:", len(summary_tokens))
    print("Summary:", summary)

def train_model(data):
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
    num_steps = float(num_samples) * num_train_epochs / (batch_size * gradient_accumulation_steps)
    steps_per_epoch = int(num_steps / num_train_epochs)

    #Load the longformer from huggingface
    led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)    #Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")    #Tokenize the data
    tokenized_data = tokenizer(data, return_tensors='pt')
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        num_train_epochs = num_train_epochs,
        fp16=False,
    #     fp16_backend="apex",
        output_dir="./",
        logging_steps=steps_per_epoch, 
        eval_steps=steps_per_epoch, 
        save_steps=steps_per_epoch, 
        warmup_steps=512,
        save_total_limit=2,
        gradient_accumulation_steps = gradient_accumulation_steps, 
        optim= "adafactor"
    )

    led.config.num_beams = 2
    led.config.max_length = 1024
    led.config.min_length = 768
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

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
    trainer.save_model("check/")


def removeScriptWords(text):
    """
    Remove words that are commonly found in scripts
    """
    #Remove script stop words
    for word in SCRIPT_STOP_WORDS:
        text = text.replace(word, '')
    return text

if __name__ == "__main__":
    #Read clean script
    with open('clean_scripts/John-Wick.txt', 'r') as file:
        data = file.read()
        #Split data into scenes
        #Each scene begins with a line that starts with a number
        scenes = data.split('\n')
        #Remove empty lines
        scenes = [line for line in scenes if line.strip() != '']
        #Find the indexes of all lines that start with a number
        scene_indexes = [i for i, line in enumerate(scenes) if line[0].isdigit()]
        #Split the data into scenes
        scenes = [scenes[i:j] for i, j in zip(scene_indexes, scene_indexes[1:]+[None])]
        print(len(scenes))
        print(scenes[0])
        #combine each scene into one string
        scenes = [' '.join(scene) for scene in scenes]
        summarize_text(removeScriptWords(data.replace('\n', ' ')))