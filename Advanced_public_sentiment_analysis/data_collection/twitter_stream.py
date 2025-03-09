"""
Twitter Stream Processor for Government-Grade Sentiment Analysis
Version: 2.0
Last Updated: 2025-03-09
Author: AI Development Team
Security Level: RESTRICTED
"""

import asyncio
import tweepy
import logging
import ssl
import json
from datetime import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from cryptography.fernet import Fernet
from redis import Redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram
from opentelemetry import trace
from .models import Tweet, DataSource
from .config import TwitterConfig, SecurityConfig
from .security import DataEncryptor, AccessControl

# Metrics and Monitoring
TWEETS_PROCESSED = Counter('tweets_processed_total', 'Total tweets processed')
PROCESSING_TIME = Histogram('tweet_processing_seconds', 'Time spent processing tweets')

class SecureStreamListener(tweepy.StreamingClient, ABC):
    """
    Enhanced Twitter Stream Listener with security and monitoring capabilities.
    Implements government-grade security standards and data handling.
    """
    
    def __init__(self, bearer_token: str, encryption_key: bytes,
                 redis_client: Redis, db_session: sessionmaker):
        super().__init__(bearer_token)
        self.encryptor = DataEncryptor(encryption_key)
        self.redis = redis_client
        self.db = db_session
        self.logger = logging.getLogger(__name__)
        self.tracer = trace.get_tracer(__name__)
        
    async def on_tweet(self, tweet: tweepy.Tweet) -> None:
        """
        Process incoming tweets with enhanced security and monitoring.
        Implements rate limiting and data validation.
        """
        with self.tracer.start_as_current_span("process_tweet") as span:
            try:
                # Data validation and sanitization
                if not self._validate_tweet(tweet):
                    self.logger.warning(f"Invalid tweet detected: {tweet.id}")
                    return

                # Encrypt sensitive data
                encrypted_text = self.encryptor.encrypt(tweet.text)
                
                # Store in Redis cache for real-time analysis
                await self._cache_tweet(tweet.id, encrypted_text)
                
                # Persist to secure database
                await self._store_tweet(tweet)
                
                # Update metrics
                TWEETS_PROCESSED.inc()
                
            except Exception as e:
                self.logger.error(f"Error processing tweet: {str(e)}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
    
    async def _cache_tweet(self, tweet_id: int, encrypted_text: bytes) -> None:
        """Cache encrypted tweet data for real-time analysis."""
        await self.redis.setex(
            f"tweet:{tweet_id}",
            TwitterConfig.CACHE_EXPIRY,
            encrypted_text
        )
    
    async def _store_tweet(self, tweet: tweepy.Tweet) -> None:
        """Securely store tweet data in the database."""
        with self.db() as session:
            tweet_record = Tweet(
                id=tweet.id,
                text=self.encryptor.encrypt(tweet.text),
                timestamp=datetime.utcnow(),
                source=DataSource.TWITTER,
                metadata=self._extract_metadata(tweet)
            )
            session.add(tweet_record)
            await session.commit()
    
    def _validate_tweet(self, tweet: tweepy.Tweet) -> bool:
        """
        Validate tweet content against security policies.
        Implements content filtering and security checks.
        """
        if not tweet.text:
            return False
        
        # Check for malicious content
        if SecurityConfig.CONTENT_FILTER.contains_malicious_content(tweet.text):
            self.logger.warning(f"Malicious content detected in tweet {tweet.id}")
            return False
            
        return True
    
    @staticmethod
    def _extract_metadata(tweet: tweepy.Tweet) -> Dict:
        """Extract and structure tweet metadata for analysis."""
        return {
            "user_id": tweet.author_id,
            "lang": tweet.lang,
            "geo": tweet.geo,
            "context_annotations": tweet.context_annotations,
            "collected_at": datetime.utcnow().isoformat()
        }

class TwitterStreamManager:
    """
    Manages Twitter stream connections with failover and load balancing.
    Implements government security standards for data collection.
    """
    
    def __init__(self):
        self.config = TwitterConfig()
        self.security = SecurityConfig()
        self.streams: List[SecureStreamListener] = []
        
    async def initialize_streams(self) -> None:
        """Initialize multiple stream listeners for redundancy."""
        for token in self.config.BEARER_TOKENS:
            stream = SecureStreamListener(
                bearer_token=token,
                encryption_key=self.security.ENCRYPTION_KEY,
                redis_client=self.config.REDIS_CLIENT,
                db_session=self.config.DB_SESSION
            )
            self.streams.append(stream)
            
    async def start_streaming(self) -> None:
        """Start streaming with automatic failover."""
        rules = [
            {"value": "government lang:en", "tag": "government"},
            {"value": "policy lang:en", "tag": "policy"},
            {"value": "election lang:en", "tag": "election"},
            {"value": "crisis lang:en", "tag": "crisis"},
            {"value": "security -is:retweet", "tag": "security"}
        ]
        
        for stream in self.streams:
            # Clear existing rules and add new ones
            await stream.delete_rules([rule.id for rule in await stream.get_rules()])
            await stream.add_rules(rules)
            
            # Start streaming with error handling
            try:
                await stream.filter(
                    tweet_fields=["context_annotations", "geo", "lang"],
                    expansions=["author_id"],
                    threaded=True
                )
            except Exception as e:
                logging.error(f"Stream error: {str(e)}")
                continue

async def main():
    """Main entry point for the Twitter stream processor."""
    manager = TwitterStreamManager()
    await manager.initialize_streams()
    await manager.start_streaming()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Start the async event loop
    asyncio.run(main())
