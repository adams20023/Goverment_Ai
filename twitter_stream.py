import requests
from flask import Flask, request, redirect
import sqlite3
import os

app = Flask(__name__)

# Replace with your client ID, client secret, and redirect URI
CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
REDIRECT_URI = 'http://localhost/callback'

# SQLite database setup
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS tweets (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT)''')

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
    auth = authenticate()
    code = request.args.get('code')
    verifier = request.args.get('verifier')
    access_token, access_token_secret = auth.get_access_token(code)
    client = tweepy.Client(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET, access_token=access_token, access_token_secret=access_token_secret)
    
    # Stream tweets
    class MyStream(tweepy.StreamingClient):
        def on_tweet(self, tweet):
            text = tweet.text
            c.execute("INSERT INTO tweets (text) VALUES (?)", (text,))
            conn.commit()
            print(f"Collected tweet: {text}")

        def on_errors(self, status_code):
            if status_code == 420:  # Rate limit reached
                return False

    # Initialize the stream
    stream = MyStream(bearer_token=client.bearer_token)

    # Add rules to filter tweets (customize keywords)
    rules = [
        tweepy.StreamRule("government"),
        tweepy.StreamRule("policy"),
        tweepy.StreamRule("election"),
        tweepy.StreamRule("manchester")
    ]

    # Add rules to the stream
    stream.add_rules(rules)

    # Start streaming
    stream.filter(tweet_fields=["text"])

    return 'Streaming started!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

conn.close()
