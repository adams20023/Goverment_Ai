import sqlite3

# Connect to database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute("ALTER TABLE analysis ADD COLUMN misinformation INTEGER DEFAULT 0")

# Rule-based detection
misinfo_keywords = ['fake', 'hoax', 'lie', 'conspiracy']
c.execute("SELECT id, text, sentiment FROM analysis")
rows = c.fetchall()

for row in rows:
    text = row[1].lower()
    is_misinfo = 1 if (row[2] == 'NEGATIVE' and any(kw in text for kw in misinfo_keywords)) else 0
    c.execute("UPDATE analysis SET misinformation = ? WHERE id = ?", (is_misinfo, row[0]))

conn.commit()
conn.close()
print("Misinformation detection complete.")
