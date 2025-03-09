import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime

# Initialize SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS news 
             (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, timestamp TEXT)''')

# Scrape BBC News
url = 'https://www.bbc.co.uk/news'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Collect headlines
for headline in soup.find_all('h3', limit=10):  # Limit to 10 for simplicity
    title = headline.text.strip()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO news (title, timestamp) VALUES (?, ?)", (title, timestamp))
    conn.commit()
    print(f"Collected news: {title}")

# Close connection
conn.close()
