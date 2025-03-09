from flask import Flask, render_template
import sqlite3
import pandas as pd

app = Flask(__name__)

@app.route('/')
def dashboard():
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query("SELECT * FROM analysis", conn)
    conn.close()
    
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    topic_counts = df['topic'].value_counts().to_dict()
    misinfo_count = df['misinformation'].sum()
    
    return render_template('dashboard.html', 
                          sentiments=sentiment_counts, 
                          topics=topic_counts, 
                          misinfo=misinfo_count)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
