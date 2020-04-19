from bs4 import BeautifulSoup
import requests
import urllib.request
import os

url = "https://www.akc.org/dog-breeds/page/2/"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

imgs = soup.findAll("img", attrs={'width': '400'})

#print(imgs)
breeds = []

for link in imgs:
    if "http" in link.get('data-src'):
        #print(link.get('data-src'))
        breeds.append(link.get('data-src'))

#print(breeds)
print(breeds[1])
img_name = os.path.basename(breeds[1])
urllib.request.urlretrieve(breeds[1], img_name)