import tarfile
import os
import sys
import re
import pickle

def clean_script(script_raw):
    #remove all html tags
    script = re.sub(r'<[^>]*>', '', script_raw)
    #remove all double new lines
    script = script.replace("\n\n", "\n")
    #remove all tabs
    script = script.replace("\t", "")
    #remove all double spaces
    script = script.replace("  ", "")  
    return script 

# load each tarfile in the scriptbase_alpha folder
table = []
for file in os.listdir("scriptbase_alpha"):
    if file.endswith(".tar.gz"):
        name = file.split(".")[0]
        tar = tarfile.open(f"scriptbase_alpha/{file}", "r:gz")
        script = ""
        summary = ""
        for method in tar.getmembers():
            if "script.txt" in method.name:
                f = tar.extractfile(method)
                if f:
                    script = clean_script(f.read().decode("utf-8", errors="ignore"))
            elif "wikiplot.txt" in method.name:
                f = tar.extractfile(method)
                if f:
                    summary = f.read().decode("utf-8")
        table.append([name, script, summary])
        tar.close()
        print(f"Loaded {name}")

# write the table to a file
with open("scriptbase_alpha_list", "wb") as f:
    pickle.dump(table, f)
