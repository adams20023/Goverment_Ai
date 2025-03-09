import sqlite3

# Connect to database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute("ALTER TABLE analysis ADD COLUMN crisis_event INTEGER DEFAULT 0")

# Crisis keywords
crisis_keywords = ['disaster', 'protest', 'cyberattack', 'riot', 'emergency']

c.execute("SELECT id, text FROM analysis")
rows = c.fetchall()

for row in rows:
    text = row[1].lower()
    is_crisis = 1 if any(kw in text for kw in crisis_keywords) else 0
    c.execute("UPDATE analysis SET crisis_event = ? WHERE id = ?", (is_crisis, row[0]))

conn.commit()
conn.close()
print("Crisis detection complete.")
