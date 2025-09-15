"""Reddit data collection service."""

import logging
import time
import random
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
import praw
from ..models import Review
from ..config import settings

logger = logging.getLogger(__name__)


class RedditService:
    """Reddit data collection service."""
    
    def __init__(self):
        self.reddit = None
        self._init_reddit()
    
    def _init_reddit(self):
        """Initialize Reddit client."""
        if (settings.reddit_client_id and 
            settings.reddit_client_secret and 
            settings.reddit_user_agent):
            try:
                self.reddit = praw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent
                )
                logger.info("Reddit client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
                self.reddit = None
        else:
            logger.warning("Reddit credentials not provided, using mock data")
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_delay, max=settings.retry_backoff * 10)
    )
    def scrape(self, query: str, limit: int = 50) -> List[Review]:
        """Scrape Reddit for reviews."""
        if self.reddit:
            return self._scrape_real(query, limit)
        else:
            return self._scrape_mock(query, limit)
    
    def _scrape_real(self, query: str, limit: int) -> List[Review]:
        """Scrape real Reddit data."""
        reviews = []
        
        try:
            # Search across multiple subreddits
            subreddits = ['technology', 'cars', 'iphone', 'android', 'gaming']
            
            for subreddit_name in subreddits:
                if len(reviews) >= limit:
                    break
                
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts
                    for submission in subreddit.search(query, limit=10):
                        if len(reviews) >= limit:
                            break
                        
                        # Get top comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:5]:  # Top 5 comments per post
                            if len(reviews) >= limit:
                                break
                            
                            if hasattr(comment, 'body') and len(comment.body) > 20:
                                review = Review(
                                    id=f"reddit_{comment.id}",
                                    platform="reddit",
                                    author=str(comment.author) if comment.author else None,
                                    created_utc=comment.created_utc,
                                    text=comment.body,
                                    url=f"https://reddit.com{submission.permalink}",
                                    meta={
                                        'subreddit': subreddit_name,
                                        'post_title': submission.title,
                                        'score': comment.score
                                    }
                                )
                                reviews.append(review)
                        
                        # Polite sleep
                        time.sleep(0.2)
                
                except Exception as e:
                    logger.warning(f"Failed to scrape subreddit {subreddit_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Reddit scraping failed: {e}")
            return self._scrape_mock(query, limit)
        
        logger.info(f"Scraped {len(reviews)} reviews from Reddit")
        return reviews
    
    def _scrape_mock(self, query: str, limit: int) -> List[Review]:
        """Generate comprehensive mock reviews for testing."""
        # Generate more detailed and varied reviews
        base_reviews = [
            f"I've been using the {query} for 6 months now and I'm absolutely blown away by its performance. The build quality is exceptional and it handles everything I throw at it without breaking a sweat. The battery life is outstanding - I can go a full day of heavy usage without worrying about charging. The camera quality is phenomenal, especially in low light conditions. The user interface is intuitive and responsive. The only minor issue I've noticed is that it can get a bit warm during intensive tasks, but it's never been a problem. Overall, this is one of the best purchases I've made in years. Highly recommend to anyone looking for a premium experience.",
            
            f"After extensive research and comparison, I decided to go with the {query}. The initial setup was smooth and the learning curve wasn't too steep. The design is sleek and modern, definitely stands out from the competition. Performance-wise, it's been solid for my daily tasks. The display quality is crisp and vibrant, perfect for both work and entertainment. However, I've encountered some software bugs that occasionally affect the user experience. The customer support has been helpful but the response time could be better. The price point is on the higher side, but considering the features and build quality, it's justified. Would recommend with some reservations about the software stability.",
            
            f"I had high expectations for the {query} based on the marketing hype, but unfortunately, it didn't live up to them. The build quality feels cheap compared to other products in the same price range. The performance is inconsistent - sometimes it's fast and responsive, other times it lags significantly. The battery life is disappointing, requiring multiple charges throughout the day. The camera quality is mediocre at best, especially in challenging lighting conditions. The user interface feels clunky and outdated. Customer service has been unresponsive to my concerns. For the price they're asking, I expected much better quality and support. I regret this purchase and would not recommend it to others.",
            
            f"The {query} has been a game-changer for my workflow. The processing power is incredible - I can run multiple demanding applications simultaneously without any performance issues. The storage capacity is generous and the transfer speeds are lightning fast. The connectivity options are comprehensive, making it easy to integrate with my existing setup. The audio quality is surprisingly good for built-in speakers. The keyboard and trackpad are comfortable for long typing sessions. The only downside is the weight - it's heavier than I expected, which affects portability. The price is steep, but the productivity gains have more than justified the investment. This is definitely a professional-grade device.",
            
            f"I've been using the {query} for about 3 months now and it's been a mixed experience. The initial setup was straightforward and the interface is clean and modern. The performance is generally good for everyday tasks, though it can struggle with more demanding applications. The battery life is decent - I can get through a typical workday without charging. The camera quality is good in well-lit conditions but struggles in low light. The build quality feels solid but not premium. The software updates have been regular, which is a plus. However, I've noticed some heating issues during extended use. The price is reasonable for what you get, but there are better options available in the market. It's an okay product but not exceptional.",
            
            f"After months of deliberation, I finally purchased the {query} and I couldn't be happier with my decision. The attention to detail in the design is remarkable - every aspect feels carefully considered. The performance exceeds my expectations, handling complex tasks with ease. The battery life is exceptional, easily lasting through multiple days of moderate usage. The camera system is outstanding, producing professional-quality photos and videos. The user experience is smooth and intuitive, with thoughtful features that enhance productivity. The build quality is top-notch, feeling premium and durable. The customer support has been excellent, with quick responses and helpful solutions. While the price is high, the quality and features justify the investment. This is a premium product that delivers on its promises.",
            
            f"I was initially excited about the {query} but my experience has been disappointing. The performance is inconsistent - sometimes it works perfectly, other times it's frustratingly slow. The battery life is poor, requiring frequent charging throughout the day. The camera quality is underwhelming, especially compared to competitors in the same price range. The user interface feels outdated and clunky. I've encountered several software bugs that affect daily usage. The build quality doesn't feel premium despite the high price tag. Customer service has been unhelpful and slow to respond. The overall experience has been frustrating and doesn't justify the cost. I would not recommend this product to others.",
            
            f"The {query} has exceeded all my expectations. The build quality is exceptional, with attention to detail that's immediately apparent. The performance is outstanding - it handles everything I throw at it without breaking a sweat. The battery life is impressive, easily lasting through a full day of heavy usage. The camera system is phenomenal, producing stunning photos and videos in various conditions. The user interface is intuitive and responsive, making daily tasks enjoyable. The connectivity options are comprehensive and reliable. The audio quality is excellent for both music and calls. The software updates are regular and bring meaningful improvements. While the price is high, the quality and features make it worth every penny. This is a premium product that delivers exceptional value.",
            
            f"I've been using the {query} for several months and it's been a solid performer overall. The design is clean and modern, though not particularly distinctive. The performance is good for most tasks, though it can struggle with intensive applications. The battery life is adequate for daily use but requires charging by evening. The camera quality is decent but not exceptional. The user interface is functional but could be more intuitive. The build quality feels sturdy but not premium. The software updates have been regular, which is appreciated. The price is reasonable for the features offered. It's a good product that does what it's supposed to do without being exceptional. I would recommend it for users who prioritize reliability over cutting-edge features.",
            
            f"My experience with the {query} has been overwhelmingly positive. The attention to detail in both hardware and software is remarkable. The performance is consistently excellent, handling demanding tasks with ease. The battery life is outstanding, often lasting multiple days with moderate usage. The camera system is exceptional, producing professional-quality results. The user experience is smooth and intuitive, with thoughtful design choices throughout. The build quality is premium, feeling solid and well-crafted. The connectivity is reliable and fast. The audio quality is impressive for built-in speakers. The customer support has been responsive and helpful. While the price is high, the quality and features justify the investment. This is a premium product that delivers on every front."
        ]
        
        # Generate additional reviews by varying the base reviews
        reviews = []
        for i in range(min(limit, 50)):  # Generate up to 50 reviews
            base_review = base_reviews[i % len(base_reviews)]
            
            # Add some variation to make reviews more diverse
            variations = [
                f"Update: {base_review}",
                f"After 2 years of use: {base_review}",
                f"Long-term review: {base_review}",
                f"Final verdict: {base_review}",
                base_review
            ]
            
            text = variations[i % len(variations)]
            
            review = Review(
                id=f"mock_{i}",
                platform="mock",
                author=f"user_{i}",
                created_utc=time.time() - random.randint(0, 86400 * 365),  # Last year
                text=text,
                url=f"https://reddit.com/r/reviews/comments/mock_{i}",
                meta={'query': query, 'length': len(text)}
            )
            reviews.append(review)
        
        logger.info(f"Generated {len(reviews)} comprehensive mock reviews")
        return reviews
