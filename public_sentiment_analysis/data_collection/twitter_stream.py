import tweepy
import sqlite3
import os
from flask import Flask, request, redirect
import webbrowser

app = Flask(__name__)

# Replace with your client ID, client secret, and redirect URI
CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
REDIRECT_URI = 'http://localhost:5000/callback'

# Twitter API credentials (replace with your own)
CONSUMER_KEY = os.environ['CONSUMER_KEY']
CONSUMER_SECRET = os.environ['CONSUMER_SECRET']

# SQLite database setup
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS tweets 
             (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, timestamp TEXT)''')

# Function to handle the OAuth2.0 authentication
def authenticate():
    auth = tweepy.OAuth2UserHandler(CONSUMER_KEY, CONSUMER_SECRET, redirect_uri=REDIRECT_URI)
    auth_url = auth.get_authorization_url()
    print(f"Please visit this URL to authorize the application: {auth_url}")
    webbrowser.open(auth_url)
    return auth

@app.route('/')
def home():
    auth = authenticate()
    return redirect(auth.get_authorization_url())

@app.route('/callback')
def callback():
    auth = tweepy.OAuth2UserHandler(CONSUMER_KEY, CONSUMER_SECRET, redirect_uri=REDIRECT_URI)
    code = request.args.get('code')
    access_token = auth.get_access_token(code)
    client = tweepy.Client(bearer_token=access_token)

    # Stream tweets
    class MyStream(tweepy.StreamingClient):
        def on_tweet(self, tweet):
            text = tweet.text
            timestamp = tweet.created_at.isoformat()
            c.execute("INSERT INTO tweets (text, timestamp) VALUES (?, ?)", (text, timestamp))
            conn.commit()
            print(f"Collected tweet: {text}")

        def on_errors(self, status_code):
            if status_code == 420:  # Rate limit reached
                print("Rate limit reached. Stopping stream.")
                return False

    # Initialize the stream
    stream = MyStream(bearer_token=access_token)

    # Add rules to filter tweets (customize keywords)
    rules = [
        tweepy.StreamRule("government"),
        tweepy.StreamRule("policy"),
        tweepy.StreamRule("election"),
        tweepy.StreamRule("crisis"),
        tweepy.StreamRule("protest"),
        tweepy.StreamRule("cyberattack"),
        tweepy.StreamRule("economy"),
        tweepy.StreamRule("security")
    ]

    # Add rules to the stream
    stream.add_rules(rules)

    # Start streaming
    stream.filter(tweet_fields=["text", "created_at"])

    return 'Streaming started!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

conn.close()
