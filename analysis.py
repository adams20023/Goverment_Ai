from transformers import pipeline
from gensim import corpora, models
import sqlite3

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Connect to database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS analysis
             (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, sentiment TEXT, topic INTEGER)''')

# Fetch all texts
c.execute("SELECT text FROM tweets UNION SELECT title FROM news")
texts = [row[0] for row in c.fetchall() if row[0]]  # Exclude empty rows

# Sentiment analysis
for text in texts:
    result = sentiment_pipeline(text[:512])[0]  # Truncate to 512 chars (model limit)
    sentiment = result['label']
    c.execute("INSERT INTO analysis (text, sentiment) VALUES (?, ?)", (text, sentiment))

# Topic modeling
texts_tokenized = [text.split() for text in texts]
dictionary = corpora.Dictionary(texts_tokenized)
corpus = [dictionary.doc2bow(text) for text in texts_tokenized]
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Assign topics
c.execute("SELECT id, text FROM analysis")
rows = c.fetchall()
for row in rows:
    bow = dictionary.doc2bow(row[1].split())
    if bow:  # Check if bow is not empty
        topic = lda_model.get_document_topics(bow)[0][0]
        c.execute("UPDATE analysis SET topic = ? WHERE id = ?", (topic, row[0]))

conn.commit()
conn.close()
print("Analysis complete.")
