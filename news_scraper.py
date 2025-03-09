import requests
from bs4 import BeautifulSoup
import sqlite3

# Set up SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS news (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT)''')

# Scrape news from a sample site (e.g., BBC)
url = 'https://www.bbc.co.uk/news'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

for headline in soup.find_all('h3', limit=10):  # Limit to 10 headlines
    title = headline.text.strip()
    c.execute("INSERT INTO news (title) VALUES (?)", (title,))
    conn.commit()
    print(f"Collected news: {title}")

conn.close()
