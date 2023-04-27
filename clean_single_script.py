from clean_script import cleanSingleScript
import sys

if __name__ == "__main__":
    #Get first cmd line argument
    file = sys.argv[1]
    print("Cleaning " + file)
    cleanSingleScript(file)