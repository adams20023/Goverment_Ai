import sqlite3

# Connect to database
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute("ALTER TABLE analysis ADD COLUMN threat_level TEXT DEFAULT 'low'")

# Threat categories and keywords
threat_keywords = {
    'terrorism': ['terrorism', 'bomb', 'attack'],
    'cybercrime': ['hack', 'phish', 'malware'],
    'illegal_activity': ['drug', 'smuggle', 'fraud']
}

c.execute("SELECT id, text FROM analysis")
rows = c.fetchall()

for row in rows:
    text = row[1].lower()
    for threat_type, keywords in threat_keywords.items():
        if any(kw in text for kw in keywords):
            c.execute("UPDATE analysis SET threat_level = ? WHERE id = ?", (threat_type, row[0]))
            break

conn.commit()
conn.close()
print("Threat monitoring complete.")
