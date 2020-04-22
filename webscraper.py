#on line 32, change the path so it works on your computer
from bs4 import BeautifulSoup
import requests
import os
import urllib.request

print("Fetching your dog images...")

breeds = []
imgs = []

url2 = "https://www.akc.org/dog-breeds/"
html2 = requests.get(url2).text
soup2 = BeautifulSoup(html2, "html.parser")
imgs2 = soup2.findAll("img", attrs={'width': '400'})
for link in imgs2:
    if "http" in link.get('data-src'):
        breeds.append(link.get('data-src'))

for x in range(2, 24):
    string = str(x)
    url = "https://www.akc.org/dog-breeds/page/" + string + "/"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    imgs = soup.findAll("img", attrs={'width': '400'})
    for link in imgs:
        if "http" in link.get('data-src'):
            breeds.append(link.get('data-src'))

for breed in breeds:
    img_name = os.path.basename(breed)
    os.chdir("/Users/emilymadril/PycharmProjects/PythonProject/dog_images")
    urllib.request.urlretrieve(breed, img_name)

print("All done!")
