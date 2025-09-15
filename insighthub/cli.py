"""Command-line interface for InsightHub."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import settings, ensure_config_files
from .models import Review
from .services.reddit_client import RedditService
from .services.llm import LLMServiceFactory
from .analysis.sentiment import VADERSentimentAnalyzer
from .analysis.aspect import YAMLAspectDetector
from .analysis.scoring import create_analysis_summary, compute_aspect_scores
from .reporting.data_prep import prepare_export, export_to_json

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_scrape(args):
    """Scrape command."""
    ensure_config_files()
    
    reddit_service = RedditService()
    reviews = reddit_service.scrape(args.query, args.limit)
    
    print(f"Scraped {len(reviews)} reviews for '{args.query}'")
    
    # Show sample
    if reviews:
        print("\nSample review:")
        sample = reviews[0]
        print(f"Author: {sample.author}")
        print(f"Text: {sample.text[:100]}...")
        print(f"Platform: {sample.platform}")


def cmd_analyze(args):
    """Analyze command."""
    ensure_config_files()
    
    # Initialize services
    reddit_service = RedditService()
    llm_service = LLMServiceFactory.create()
    sentiment_analyzer = VADERSentimentAnalyzer()
    aspect_detector = YAMLAspectDetector()
    
    print(f"Analyzing '{args.query}' with limit {args.limit}...")
    
    # Scrape reviews
    reviews = reddit_service.scrape(args.query, args.limit)
    print(f"Scraped {len(reviews)} reviews")
    
    if not reviews:
        print("No reviews found!")
        return
    
    # Analyze sentiment
    sentiment_results = []
    for review in reviews:
        try:
            result = sentiment_analyzer.analyze(review.text)
            sentiment_results.append(result)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for review {review.id}: {e}")
            sentiment_results.append({"compound": 0.0, "label": "NEUTRAL", "stars": 3.0})
    
    # Count sentiments
    pos_count = sum(1 for r in sentiment_results if r.get('label') == 'POSITIVE')
    neg_count = sum(1 for r in sentiment_results if r.get('label') == 'NEGATIVE')
    neu_count = len(sentiment_results) - pos_count - neg_count
    
    print(f"Sentiment distribution: {pos_count} positive, {neg_count} negative, {neu_count} neutral")
    
    # Detect aspects
    aspect_reviews = {}
    for review, sentiment in zip(reviews, sentiment_results):
        aspects = aspect_detector.detect_aspects(review.text)
        for aspect in aspects:
            if aspect not in aspect_reviews:
                aspect_reviews[aspect] = []
            aspect_reviews[aspect].append(sentiment)
    
    # Compute aspect scores
    aspect_scores = compute_aspect_scores(aspect_reviews)
    
    # Create summary
    summary = create_analysis_summary(args.query, reviews, sentiment_results, aspect_scores)
    
    # Calculate baseline for logging
    baseline = 1.0 + 4.0 * ((pos_count + 0.5 * neu_count) / len(reviews))
    print(f"Baseline: {baseline:.2f}, Final average: {summary.avg_stars:.2f}")
    
    # Generate pros/cons
    pros_cons = llm_service.generate_pros_cons(reviews, args.query)
    
    # Export results
    if args.out:
        export_data = prepare_export(reviews, summary, pros_cons)
        export_to_json(export_data, args.out)
        print(f"Results exported to {args.out}")
    
    # Print summary
    print(f"\nAnalysis Summary for '{args.query}':")
    print(f"Total reviews: {summary.total}")
    print(f"Average rating: {summary.avg_stars:.2f}/5")
    print(f"Positive: {summary.pos} ({summary.pos/summary.total*100:.1f}%)")
    print(f"Negative: {summary.neg} ({summary.neg/summary.total*100:.1f}%)")
    print(f"Neutral: {summary.neu} ({summary.neu/summary.total*100:.1f}%)")
    
    if aspect_scores:
        print("\nAspect scores:")
        for aspect, score in aspect_scores.items():
            print(f"  {aspect}: {score:.2f}/5")
    
    print(f"\nPros: {pros_cons['pros']}")
    print(f"Cons: {pros_cons['cons']}")


def cmd_ui(args):
    """UI command."""
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"Streamlit app not found at {app_path}")
        return
    
    print("Launching InsightHub UI...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch UI: {e}")
    except KeyboardInterrupt:
        print("\nUI stopped by user")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="InsightHub - Cross-Platform Review Intelligence")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape reviews')
    scrape_parser.add_argument('query', help='Search query')
    scrape_parser.add_argument('--limit', type=int, default=50, help='Number of reviews to scrape')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze reviews')
    analyze_parser.add_argument('query', help='Search query')
    analyze_parser.add_argument('--limit', type=int, default=50, help='Number of reviews to analyze')
    analyze_parser.add_argument('--out', help='Output JSON file')
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch web UI')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging()
    
    try:
        if args.command == 'scrape':
            cmd_scrape(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'ui':
            cmd_ui(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)
