# webscraper code

#from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import requests

#driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
#breeds=[]


URL = 'https://www.akc.org/dog-breeds/'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')

breed = soup.find(id="breed-type-card__title mt0 mb0 f-25 py3 px3")