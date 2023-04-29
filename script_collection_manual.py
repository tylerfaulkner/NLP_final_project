from script_collection_auto import getScript
import sys

def main(argv):
    #Get the URL from the command line
    url = argv[0]
    save_folder = argv[1]
    getScript(url, save_folder)

if __name__ == "__main__":
    main(sys.argv[1:])