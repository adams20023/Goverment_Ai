"""
Advanced News Scraping System for Government Intelligence
Version: 2.1
Last Updated: 2025-03-09 03:15:43
Author: adams20023
Security Classification: RESTRICTED
"""

import asyncio
import aiohttp
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
from bs4 import BeautifulSoup
from prometheus_client import Counter, Gauge, Histogram
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession
from cryptography.fernet import Fernet
from ratelimit import limits, sleep_and_retry
from .models import NewsArticle, DataSource
from .config import NewsConfig, SecurityConfig
from .security import DataEncryptor, AccessControl

# Telemetry setup
ARTICLES_SCRAPED = Counter('news_articles_scraped_total', 'Total articles scraped')
SCRAPING_TIME = Histogram('news_scraping_seconds', 'Time spent scraping articles')
ACTIVE_SOURCES = Gauge('news_sources_active', 'Number of active news sources')

class NewsSource:
    """Configuration for trusted news sources with specific parsing rules."""
    def __init__(self, name: str, url: str, selectors: Dict[str, str], 
                 trust_score: float = 1.0):
        self.name = name
        self.url = url
        self.selectors = selectors
        self.trust_score = trust_score
        self.last_fetch = None
        self.articles_count = 0

class SecureNewsScraper:
    """
    Enhanced news scraping system with government-grade security and verification.
    Implements content validation, source verification, and secure storage.
    """
    
    def __init__(self, db_session: AsyncSession, encryption_key: bytes):
        self.db = db_session
        self.encryptor = DataEncryptor(encryption_key)
        self.logger = logging.getLogger(__name__)
        self.tracer = trace.get_tracer(__name__)
        self.seen_articles: Set[str] = set()
        
        # Initialize trusted news sources
        self.sources = {
            'bbc': NewsSource(
                name='BBC News',
                url='https://www.bbc.co.uk/news',
                selectors={
                    'headline': 'h3.gs-c-promo-heading',
                    'content': 'article.gs-c-promo',
                    'timestamp': 'time.gs-o-timestamp'
                },
                trust_score=0.95
            ),
            'reuters': NewsSource(
                name='Reuters',
                url='https://www.reuters.com',
                selectors={
                    'headline': 'h3.article-heading',
                    'content': 'article.story',
                    'timestamp': 'time.article-time'
                },
                trust_score=0.98
            )
            # Add more trusted sources as needed
        }
        
        ACTIVE_SOURCES.set(len(self.sources))

    @sleep_and_retry
    @limits(calls=100, period=60)  # Rate limiting
    async def fetch_articles(self) -> None:
        """
        Fetch articles from trusted sources with advanced error handling
        and validation.
        """
        async with aiohttp.ClientSession() as session:
            fetch_tasks = []
            for source in self.sources.values():
                task = asyncio.create_task(
                    self._fetch_from_source(session, source)
                )
                fetch_tasks.append(task)
            
            await asyncio.gather(*fetch_tasks)

    async def _fetch_from_source(self, 
                               session: aiohttp.ClientSession, 
                               source: NewsSource) -> None:
        """Fetch and process articles from a specific source."""
        with self.tracer.start_as_current_span(f"fetch_{source.name}") as span:
            try:
                async with session.get(source.url, ssl=True) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"Failed to fetch from {source.name}: {response.status}"
                        )
                        return

                    html = await response.text()
                    articles = await self._parse_articles(html, source)
                    
                    # Process and store articles
                    for article in articles:
                        if await self._validate_article(article):
                            await self._store_article(article, source)
                            ARTICLES_SCRAPED.inc()

            except Exception as e:
                self.logger.error(f"Error processing {source.name}: {str(e)}")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

    async def _parse_articles(self, 
                            html: str, 
                            source: NewsSource) -> List[Dict]:
        """Parse articles with advanced content extraction and validation."""
        soup = BeautifulSoup(html, 'html.parser')
        articles = []

        for element in soup.select(source.selectors['content']):
            try:
                headline = element.select_one(source.selectors['headline'])
                timestamp_elem = element.select_one(source.selectors['timestamp'])
                
                if headline and headline.text:
                    article = {
                        'title': headline.text.strip(),
                        'url': headline.get('href', ''),
                        'source': source.name,
                        'timestamp': self._parse_timestamp(timestamp_elem),
                        'trust_score': source.trust_score,
                        'metadata': {
                            'collected_by': 'adams20023',
                            'collected_at': datetime.utcnow().isoformat(),
                            'source_url': source.url
                        }
                    }
                    articles.append(article)
            
            except Exception as e:
                self.logger.warning(f"Error parsing article: {str(e)}")
                continue

        return articles

    async def _validate_article(self, article: Dict) -> bool:
        """
        Validate article content against security policies and
        content guidelines.
        """
        # Check for duplicate content
        content_hash = hash(article['title'])
        if content_hash in self.seen_articles:
            return False
        
        # Validate content against security policies
        if not SecurityConfig.CONTENT_FILTER.is_safe_content(article['title']):
            self.logger.warning(f"Unsafe content detected: {article['title']}")
            return False
        
        self.seen_articles.add(content_hash)
        return True

    async def _store_article(self, 
                           article: Dict, 
                           source: NewsSource) -> None:
        """Securely store article data with encryption."""
        try:
            encrypted_title = self.encryptor.encrypt(article['title'])
            
            news_article = NewsArticle(
                title=encrypted_title,
                url=article['url'],
                source=source.name,
                trust_score=source.trust_score,
                timestamp=article['timestamp'],
                metadata=json.dumps(article['metadata'])
            )
            
            self.db.add(news_article)
            await self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing article: {str(e)}")
            await self.db.rollback()

    @staticmethod
    def _parse_timestamp(timestamp_elem) -> datetime:
        """Parse and validate article timestamps."""
        if timestamp_elem and timestamp_elem.get('datetime'):
            try:
                return datetime.fromisoformat(timestamp_elem['datetime'])
            except ValueError:
                pass
        return datetime.utcnow()

async def main():
    """Main entry point for the news scraping system."""
    # Initialize database session and encryption
    db_session = AsyncSession(bind=engine)
    encryption_key = SecurityConfig.get_encryption_key()
    
    # Initialize and run scraper
    scraper = SecureNewsScraper(db_session, encryption_key)
    
    while True:
        try:
            await scraper.fetch_articles()
            await asyncio.sleep(NewsConfig.FETCH_INTERVAL)
        except Exception as e:
            logging.error(f"Main loop error: {str(e)}")
            await asyncio.sleep(60)  # Error cooldown

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Start the async event loop
    asyncio.run(main())
