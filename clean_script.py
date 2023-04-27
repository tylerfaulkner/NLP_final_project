import os
import re
def clean():
    #For each file in raw_scripts
    for file in os.listdir("raw_scripts"):
        #clean the file
        cleanSingleScript(file)

def cleanSingleScript(file):
    with open("raw_scripts/" + file, "r") as f:
        text = f.readlines()
        #strip each line
        text = [line.strip() for line in text]
        text = "\n".join(text)
        #remove all double new lines
        text = text.replace("\n\n", "\n")
        #Get word count
        word_count = len(re.findall(r'\w+', text))
        print(file + " - " + str(word_count))
        #save the file
        with open("clean_scripts/" + file, "w") as f:
            f.write(text)

if __name__ == "__main__":
    clean()

