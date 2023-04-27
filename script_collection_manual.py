from script_collection_auto import getScript
import sys

def main(argv):
    #Get the URL from the command line
    url = argv[0]
    getScript(url)

if __name__ == "__main__":
    main(sys.argv[1:])