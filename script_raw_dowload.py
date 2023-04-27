from bs4 import BeautifulSoup
import requests
import sys
import os

if __name__ == "__main__":
    #Get the URL from the command line
    url = sys.argv[1]
    #Use the requests library to get the HTML from the URL
    response = requests.get(url)
    #Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    td = soup.find("td", class_="scrtext")
    #Get the child pre tag
    pre = td.findChild("pre")
    #Save the text to a file
    #Get the name of the file from the URL before the .html
    filename = url.split("/")[-1].split(".")[0]
    text = pre.text
    #Removed \ufffd characters
    text = text.replace("\ufffd", "")
    #Try to delete the file if it exists
    try:
        os.remove(filename + ".txt")
    except OSError:
        pass
    with open(filename + ".txt", "w") as f:
        f.write(text)