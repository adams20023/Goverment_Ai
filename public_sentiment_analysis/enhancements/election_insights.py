import sqlite3

# Connect to database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute("ALTER TABLE analysis ADD COLUMN election_insight TEXT DEFAULT ''")

# Election keywords
election_keywords = ['election', 'vote', 'candidate', 'poll']

c.execute("SELECT id, text, sentiment FROM analysis")
rows = c.fetchall()

for row in rows:
    text = row[1].lower()
    if any(kw in text for kw in election_keywords):
        insight = f"Sentiment: {row[2]}"
        c.execute("UPDATE analysis SET election_insight = ? WHERE id = ?", (insight, row[0]))

conn.commit()
conn.close()
print("Election insights complete.")
