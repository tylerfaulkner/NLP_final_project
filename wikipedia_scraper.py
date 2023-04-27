
# import required modules
from bs4 import BeautifulSoup
import requests

movies = ['Die Hard', 'Terminator 2: Judgment Day', 'The Matrix', 'John Wick (film)', 'Lethal Weapon', 'Mad Max: Fury Road', 'The Bourne Identity (2002 film)', 'Predator (film)', 'The Raid: Redemption', 'The Transporter', 'Point Break', 'Speed (1994 film)', 'Taken (film)', 'Rambo: First Blood',
          'The Expendables (2010 film)', 'Face/Off', 'RoboCop', 'True Lies', 'The Terminator', 'Total Recall (1990 film)', 'Kill Bill: Vol. 1', 'The Equalizer (film)', 'The Raid 2', 'Hard Boiled', 'The Rock (film)', 'Con Air', 'The Punisher (2004 film)', 'Atomic Blonde', 'Rambo (2008 film)', 'Extraction (2020 film)']
for movie in movies:
    page = requests.get("https://en.wikipedia.org/wiki/" +
                        movie.replace(' ', '_'))
    soup = BeautifulSoup(page.content, 'html.parser')
    headers = soup.find_all('h2')
    start, end = headers[1], headers[2]

    plot = ''
    item = start.nextSibling

    while item != end:
        plot += str(item)
        item = item.nextSibling

    plot = BeautifulSoup(plot, "html.parser").text


    filename = movie if movie != 'Face/Off' else 'Face Off'
    f = open('summaries/' + filename.replace(':', '') +
            '.txt', 'a+', encoding='utf-8')
    f.write(plot.strip())
    f.close()
    print(movie)

print('done')
