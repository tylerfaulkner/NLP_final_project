import requests
import sys, getopt
import time
import os
from bs4 import BeautifulSoup

def main():
    movieDBURL = 'https://imsdb.com/scripts/'
    #Load the movies list
    movie_names = []
    with open('movies.txt') as f:
        movie_names = f.readlines()
    #Replace spaces on the movie names with dashes
    movies = [movie.replace(" ", "-") for movie in movie_names]
    #Remove the new line character
    movies = [movie.replace("\n", "") for movie in movies]
    #Remove Colons
    movies = [movie.replace(":", "") for movie in movies]
    #Remove periods
    movies = [movie.replace(".", "") for movie in movies]
    #Replace / with -
    movies = [movie.replace("/", "-") for movie in movies]
    #Move The to the end of the name
    movies = [movie.replace("The-", "") + ",-The" if movie.startswith("The-") else movie for movie in movies]
    #Add the URL to the movie name
    movies = [movieDBURL + movie + ".html" for movie in movies]
    #Get the script for each movie
    #delete the failed.txt file if it exists
    try:
        os.remove("failed.txt")
    except OSError:
        pass
    for i in range(len(movies)):
        movie = movies[i]
        movie_name = movie_names[i]
        getScript(movie, movie_name)
        #Wait 1 second to avoid overloading the server
        time.sleep(1)

def attemptAddToFailList(movie_name = None):
    if movie_name is not None:
        with open("failed.txt", "a") as f:
            f.write(movie_name)

def getScript(url, folder='raw_scripts', movie_name = None):
    #Get the HTML from the URL
    r = requests.get(url)
    #Save the movie name to a file if the reuquest failed
    if r.status_code != 200:
        print("Failed - " + str(r.status_code))
        attemptAddToFailList(movie_name)
    else:
        #Find the td with class "srctext"
        print("Parsing: " + url)
        soup = BeautifulSoup(r.text, 'html.parser')
        td = soup.find("td", class_="scrtext")
        #Get the child pre tag
        pre = td.findChild("pre")
       
        #Remove all tags from pre
        for tag in pre.find_all(True):
            tag.replaceWithChildren()

        #Mark as error if text is empty
        if pre.text == "":
            print("Failed - Empty Text")
            attemptAddToFailList(movie_name)
        else:
            #Save the text to a file
            #Get the name of the file from the URL before the .html
            filename = url.split("/")[-1].split(".")[0]
            text = pre.text
            #Removed \ufffd characters
            text = text.replace("\ufffd", "")
            #Try to delete the file if it exists
            try:
                os.remove(folder + "/"+ filename + ".txt")
            except OSError:
                pass
            with open(folder + "/"+ filename + ".txt", "w") as f:
                f.write(text)
    
if __name__ == "__main__":
   main()
