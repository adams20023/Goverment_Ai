import sqlite3

# Connect to database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute("ALTER TABLE analysis ADD COLUMN cybersecurity_alert INTEGER DEFAULT 0")

# Cybersecurity keywords
cybersecurity_keywords = ['hack', 'phish', 'malware', 'breach', 'ddos']

c.execute("SELECT id, text FROM analysis")
rows = c.fetchall()

for row in rows:
    text = row[1].lower()
    is_alert = 1 if any(kw in text for kw in cybersecurity_keywords) else 0
    c.execute("UPDATE analysis SET cybersecurity_alert = ? WHERE id = ?", (is_alert, row[0]))

conn.commit()
conn.close()
print("Cybersecurity alerts detection complete.")
