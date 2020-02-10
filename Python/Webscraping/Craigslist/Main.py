from requests import get
from bs4 import BeautifulSoup
from time import sleep
from random import randint
import pandas as pd
from datetime import date

URL = "https://washingtondc.craigslist.org/search/nva/zip?"

distance = "15" #mile
zip_num = "20120" 

post_links = []
post_dates = []
post_titles = []

#link for free stuff on craigslist
response = get( URL + 
               "search_distance=" + distance + 
               "&postal=" + zip_num +
               "&s=")

html_soup = BeautifulSoup(response.text, 'html.parser')
total_listings = int(html_soup.find(class_="totalcount").text)
    
post_scraped = 0

while(post_scraped < total_listings) :
    posts = html_soup.find_all('li', class_= 'result-row') #number of posts per page
    
    # getting info from each posting
    for i in range(0,len(posts)):     
        post = posts[i]
        
        post_dates.append (post.find('time', class_= 'result-date') ['datetime'])
        post_links.append (post.find('a', class_= 'result-image') ['href']) 
        post_titles.append (post.find('a', class_='result-title hdrlnk').text.strip() )
        
        post_scraped += 1
    
    print("Scraped: " + str(post_scraped) )
    
    if (post_scraped < total_listings):
        #slow down the request
        sleep(randint(1,5))    
    
        response = get(URL + 
                       "search_distance=" + distance + 
                       "&postal=" + zip_num +
                       "&s=" + str(post_scraped))   #parameters for next page listings
        
        html_soup = BeautifulSoup(response.text, 'html.parser')
    
#Creating pandas data frame with the info   
free_stuff_df = pd.DataFrame({'posted': post_dates,
                   'post title': post_titles,
                    'URL': post_links })
    
free_stuff_df.to_csv(r'pandas_' +str(date.today()) + '.csv', index=None, sep=',', mode='a')    
    