from flask import Flask, jsonify, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_cors import CORS
from celery import Celery
from redis import Redis
from flask_socketio import SocketIO
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import stripe
import os
from datetime import datetime, timedelta

# Initialize Flask app with production-ready configuration
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY'),
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'postgresql://localhost/sentiment_db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_TYPE='redis',
    CACHE_TYPE='redis',
    CACHE_REDIS_URL=os.environ.get('REDIS_URL'),
    STRIPE_PUBLIC_KEY=os.environ.get('STRIPE_PUBLIC_KEY'),
    STRIPE_SECRET_KEY=os.environ.get('STRIPE_SECRET_KEY')
)

# Initialize extensions
db = SQLAlchemy(app)
cache = Cache(app)
cors = CORS(app)
socketio = SocketIO(app)
Session(app)
redis_client = Redis.from_url(app.config['CACHE_REDIS_URL'])
stripe.api_key = app.config['STRIPE_SECRET_KEY']

# Configure Celery
celery = Celery(
    app.name,
    broker=os.environ.get('CELERY_BROKER_URL'),
    backend=os.environ.get('CELERY_RESULT_BACKEND')
)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    subscription_status = db.Column(db.String(50), default='free')
    api_calls = db.Column(db.Integer, default=0)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text)
    sentiment = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# Authentication middleware
def token_required(f):
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.get(data['user_id'])
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    user = User(
        email=data['email'],
        password_hash=generate_password_hash(data['password'])
    )
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()
    if user and check_password_hash(user.password_hash, data['password']):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'])
        return jsonify({'token': token})
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/sentiment', methods=['GET'])
@token_required
@cache.cached(timeout=300)
def get_sentiment_data(current_user):
    """Get cached sentiment analysis data with rate limiting"""
    if current_user.api_calls >= get_rate_limit(current_user.subscription_status):
        return jsonify({'message': 'Rate limit exceeded'}), 429
    
    current_user.api_calls += 1
    db.session.commit()
    
    results = AnalysisResult.query.filter_by(user_id=current_user.id).all()
    return jsonify([{
        'text': r.text,
        'sentiment': r.sentiment,
        'timestamp': r.timestamp
    } for r in results])

@app.route('/api/subscribe', methods=['POST'])
@token_required
def create_subscription(current_user):
    """Handle subscription payments via Stripe"""
    try:
        customer = stripe.Customer.create(
            email=current_user.email,
            source=request.json['token']
        )
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{'price': 'price_H5ggYwtDq4fbrJ'}]
        )
        current_user.subscription_status = 'premium'
        db.session.commit()
        return jsonify({'message': 'Subscription successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# WebSocket for real-time updates
@socketio.on('connect')
@token_required
def handle_connect(current_user):
    """Handle WebSocket connections"""
    session['user_id'] = current_user.id

@socketio.on('subscribe_updates')
def handle_updates():
    """Stream real-time sentiment updates"""
    while True:
        socketio.sleep(1)
        # Emit new analysis results
        new_results = get_new_results(session.get('user_id'))
        if new_results:
            socketio.emit('sentiment_update', new_results)

# Celery task for background processing
@celery.task
def process_sentiment_analysis(text):
    """Background task for sentiment analysis"""
    # Add sentiment analysis logic here
    pass

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
