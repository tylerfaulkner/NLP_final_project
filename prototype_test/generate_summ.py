import requests

url = "http://44.201.114.68:5000"

default_script = "test_data/Alien.txt"

movie_path = input("Enter the path to the movie script (example, test_data/Alien_script.txt): ")
if movie_path == "":
    movie_path = default_script

# load testset
script, summ = "", ""
with open(movie_path, "r") as f:
    script = f.read()

print("Sending script to server")
print("Response will take about 1 minute")
r = requests.get(url + "/generateSummary", data=script)
print(r.text)



