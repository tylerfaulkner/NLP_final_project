import os
import json

if __name__ == "__main__":
    #average the scores form each file in the metrics folder
    #and write the average to a file

    #get the list of files in the metrics folder
    files = os.listdir("./metrics")
    
    rougeprecision = 0
    rougerecall = 0
    rougef1 = 0

    #for each file in the metrics folder
    for file in files:
        with open("metrics/" + file, "r") as f:
            #read the fiel as a json
            lines = f.readlines()
            line = lines[0]
            line = line.replace("'", "\"")
            print(line)
            jsonLine = json.loads(line)
            #add the rougeLsum scores
            rougeprecision += jsonLine["rougeLsum_precision"]
            rougerecall += jsonLine["rougeLsum_recall"]
            rougef1 += jsonLine["rougeLsum_fmeasure"]
    #print average scores
    print("Average Precision: " + str(rougeprecision/len(files)))
    print("Average Recall: " + str(rougerecall/len(files)))
    print("Average F1: " + str(rougef1/len(files)))
    #write average scores to a file
    with open("averages.txt", "w") as f:
        f.write("Average Precision: " + str(rougeprecision/len(files)))
        f.write("Average Recall: " + str(rougerecall/len(files)))
        f.write("Average F1: " + str(rougef1/len(files)))
