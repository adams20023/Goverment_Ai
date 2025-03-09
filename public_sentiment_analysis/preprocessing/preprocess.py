import re
import sqlite3
from nltk.corpus import stopwords

# Load stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = text.lower()                  # Lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Connect to database
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Clean tweets
c.execute("SELECT id, text FROM tweets")
tweets = c.fetchall()
for tweet in tweets:
    cleaned = clean_text(tweet[1])
    c.execute("UPDATE tweets SET text = ? WHERE id = ?", (cleaned, tweet[0]))

# Clean news
c.execute("SELECT id, title FROM news")
news = c.fetchall()
for article in news:
    cleaned = clean_text(article[1])
    c.execute("UPDATE news SET title = ? WHERE id = ?", (cleaned, article[0]))

# Commit changes and close
conn.commit()
conn.close()
print("Data preprocessing complete.")
