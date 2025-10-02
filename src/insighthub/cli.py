"""Command-line interface for InsightHub."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .core.config import settings
from .core.models import Review
from .services.reddit_client import RedditService
from .services.llm import LLMServiceFactory
from .core.scoring import aggregate_generic, rank_entities
from .utils.data_prep import export_to_json
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_scrape(args):
    """Scrape command."""
    # Config files are no longer needed with GPT-only pipeline
    
    reddit_service = RedditService()
    reviews = reddit_service.scrape(args.query, args.limit)
    
    print(f"Scraped {len(reviews)} reviews for '{args.query}'")
    
    # Show sample
    if reviews:
        print("\nSample review:")
        sample = reviews[0]
        if isinstance(sample, dict):
            print(f"Author: {sample.get('author', 'Unknown')}")
            print(f"Text: {sample.get('text', '')[:100]}...")
            print(f"Source: {sample.get('source', 'Unknown')}")
        else:
            print(f"Author: {sample.author}")
            print(f"Text: {sample.text[:100]}...")
            print(f"Platform: {sample.platform}")


def cmd_analyze(args):
    """Analyze command using GPT-only pipeline."""
    # Config files are no longer needed with GPT-only pipeline
    
    # Initialize services
    reddit_service = RedditService()
    llm_service = LLMServiceFactory.create()
    
    print(f"Analyzing '{args.query}' with limit {args.limit}...")
    
    # Scrape reviews
    reviews = reddit_service.scrape(args.query, args.limit)
    print(f"Scraped {len(reviews)} reviews")
    
    if not reviews:
        print("No reviews found!")
        return
    
    # Detect intent and generate schema
    intent_schema = llm_service.detect_intent_and_schema(args.query)
    print(f"Detected intent: {intent_schema.intent}")
    if intent_schema.entity_type:
        print(f"Entity type: {intent_schema.entity_type}")
    print(f"Aspects: {intent_schema.aspects}")
    
    # Convert reviews to comment format for annotation
    comments = []
    for review in reviews:
        comment = {
            "id": review.get("id") if isinstance(review, dict) else review.id,
            "text": review.get("text") if isinstance(review, dict) else review.text,
            "upvotes": review.get("upvotes", 0) if isinstance(review, dict) else getattr(review, "upvotes", 0),
            "permalink": review.get("permalink", "") if isinstance(review, dict) else getattr(review, "permalink", "")
        }
        comments.append(comment)
    
    # Annotate comments with GPT
    print("Annotating comments with GPT...")
    annos = llm_service.annotate_comments_with_gpt(comments, intent_schema.aspects, intent_schema.entity_type, query)
    print(f"Annotated {len(annos)} comments")
    
    # Show extracted entities for verification
    if len(annos) > 0:
        all_entities = []
        for anno in annos:
            all_entities.extend(anno.entities)
        print(f"Extracted {len(all_entities)} entities")
    
    # Create upvote map for weighting
    upvote_map = {comment["id"]: comment["upvotes"] for comment in comments}
    
    # Process based on intent
    if intent_schema.intent == "RANKING":
        # Rank entities
        ranking = rank_entities(annos, upvote_map, intent_schema.entity_type, min_mentions=1)
        print(f"Found {len(ranking)} ranked entities")
        
        # Attach quotes to ranking items
        for item in ranking:
            # Find comments mentioning this entity
            entity_comments = []
            for comment in comments:
                if item.name.lower() in comment["text"].lower():
                    entity_comments.append(comment["text"][:200] + "...")
            item.quotes = entity_comments[:3]  # Top 3 quotes
        
        # Generate summary
        summary = llm_service.summarize_generic_with_gpt(
            args.query, 
            {item.name: item.overall_stars for item in ranking[:5]}, 
            sum(item.overall_stars for item in ranking[:5]) / len(ranking[:5]) if ranking else 3.0,
            [quote for item in ranking[:3] for quote in item.quotes]
        )
        
        # Prepare payload
        payload = {
            "query": args.query,
            "intent": intent_schema.intent,
            "summary": summary,
            "metadata": {"timestamp": time.time()},
            "ranking": [
                {
                    "name": item.name,
                    "overall_stars": item.overall_stars,
                    "aspect_scores": item.aspect_scores,
                    "mentions": item.mentions,
                    "confidence": item.confidence,
                    "quotes": item.quotes
                }
                for item in ranking
            ]
        }
        
    elif intent_schema.intent == "SOLUTION":
        # Group by cluster key
        clusters = defaultdict(list)
        for anno in annos:
            if anno.solution_key:
                clusters[anno.solution_key].append(anno)
        
        # Create solution clusters
        solution_clusters = []
        for cluster_key, cluster_annos in clusters.items():
            if len(cluster_annos) >= 2:  # Minimum 2 comments per cluster
                cluster = {
                    "title": cluster_key,
                    "steps": [],  # Would need additional GPT call to extract steps
                    "caveats": [],  # Would need additional GPT call to extract caveats
                    "evidence_count": len(cluster_annos)
                }
                solution_clusters.append(cluster)
        
        # Generate summary
        summary = llm_service.summarize_solutions_with_gpt(args.query, solution_clusters)
        
        # Prepare payload
        payload = {
            "query": args.query,
            "intent": intent_schema.intent,
            "summary": summary,
            "metadata": {"timestamp": time.time()},
            "solutions": solution_clusters
        }
        
    else:  # GENERIC
        # Aggregate generic results
        overall, aspect_averages = aggregate_generic(intent_schema.aspects, annos, upvote_map)
        
        # Select representative quotes
        quotes = []
        for comment in comments[:10]:  # Top 10 comments
            quotes.append(comment["text"][:200] + "...")
        
        # Generate summary
        summary = llm_service.summarize_generic_with_gpt(args.query, aspect_averages, overall, quotes)
        
        # Prepare payload
        payload = {
            "query": args.query,
            "intent": intent_schema.intent,
            "summary": summary,
            "metadata": {"timestamp": time.time()},
            "overall": overall,
            "aspects": aspect_averages,
            "quotes": quotes
        }
    
    # Export to JSON
    if args.out:
        export_to_json(payload, args.out)
        print(f"Results exported to {args.out}")
    
    # Print summary
    print(f"\nAnalysis Summary for '{args.query}':")
    print(f"Intent: {intent_schema.intent}")
    print(f"Summary: {summary[:200]}...")
    
    if intent_schema.intent == "RANKING":
        print(f"\nTop ranked entities:")
        for i, item in enumerate(ranking[:5], 1):
            print(f"  {i}. {item.name}: {item.overall_stars:.1f}/5 ({item.mentions} mentions)")
    elif intent_schema.intent == "SOLUTION":
        print(f"\nSolution clusters:")
        for i, cluster in enumerate(solution_clusters, 1):
            print(f"  {i}. {cluster['title']}: {cluster['evidence_count']} comments")
    else:
        print(f"\nOverall rating: {overall:.1f}/5")
        print(f"Aspect scores:")
        for aspect, score in aspect_averages.items():
            print(f"  {aspect}: {score:.1f}/5")


def cmd_export(args):
    """Export command."""
    import json
    
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        if args.pretty:
            # Pretty print the JSON
            print(json.dumps(data, indent=2))
        else:
            # Export to file
            output_file = args.output or args.input_file.replace('.json', '_export.json')
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Exported to {output_file}")
            
    except FileNotFoundError:
        print(f"Input file {args.input_file} not found")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in input file: {e}")
    except Exception as e:
        print(f"Export failed: {e}")


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
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export analysis results')
    export_parser.add_argument('--in', dest='input_file', required=True, help='Input JSON file')
    export_parser.add_argument('--out', dest='output', help='Output file (optional)')
    export_parser.add_argument('--pretty', action='store_true', help='Pretty print to stdout')
    
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
        elif args.command == 'export':
            cmd_export(args)
        elif args.command == 'ui':
            cmd_ui(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)
