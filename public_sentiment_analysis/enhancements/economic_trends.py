import sqlite3

# Connect to database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute("ALTER TABLE analysis ADD COLUMN economic_trend TEXT DEFAULT ''")

# Economic keywords
economic_keywords = ['economy', 'market', 'stock', 'inflation', 'gdp']

c.execute("SELECT id, text, sentiment FROM analysis")
rows = c.fetchall()

for row in rows:
    text = row[1].lower()
    if any(kw in text for kw in economic_keywords):
        trend = f"Sentiment: {row[2]}"
        c.execute("UPDATE analysis SET economic_trend = ? WHERE id = ?", (trend, row[0]))

conn.commit()
conn.close()
print("Economic trends analysis complete.")
