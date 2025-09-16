"""Data preparation for export."""

import json
from typing import Dict, Any, List
from ..core.models import Review, AnalysisSummary


def prepare_export(
    reviews: List[Review], 
    summary: AnalysisSummary, 
    pros_cons_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare data for JSON export."""
    
    # Convert reviews to serializable format
    reviews_data = []
    for review in reviews:
        if isinstance(review, dict):
            review_dict = {
                "id": review.get("id", "unknown"),
                "platform": review.get("source", "unknown"),
                "author": review.get("author", "unknown"),
                "created_utc": review.get("created_utc"),
                "text": review.get("text", ""),
                "url": review.get("url", ""),
                "meta": {"upvotes": review.get("upvotes", 0)}
            }
        else:
            review_dict = {
                "id": review.id,
                "platform": review.platform,
                "author": review.author,
                "created_utc": review.created_utc,
                "text": review.text,
                "url": review.url,
                "meta": review.meta
            }
        reviews_data.append(review_dict)
    
    # Extract pros, cons, and detailed summary from pros_cons_data
    pros = pros_cons_data.get("pros", [])
    cons = pros_cons_data.get("cons", [])
    detailed_summary = pros_cons_data.get("summary", "")
    
    # Create export data
    export_data = {
        "query": summary.query,
        "summary": {
            "total": summary.total,
            "positive": summary.pos,
            "negative": summary.neg,
            "neutral": summary.neu,
            "average_stars": summary.avg_stars,
            "aspect_averages": summary.aspect_averages,
            "detailed_summary": detailed_summary
        },
        "pros": pros,
        "cons": cons,
        "reviews": reviews_data,
        "metadata": {
            "export_timestamp": None,  # Will be set by caller
            "version": "0.1.0"
        }
    }
    
    return export_data


def build_summary_payload(query, reduce_json, reviews_by_id):
    """Build normalized summary payload for UI rendering."""
    quotes_out = []
    for q in reduce_json.get("quotes", []):
        rid = q.get("id")
        src = reviews_by_id.get(rid) if reviews_by_id else None
        if src:
            author = src.get("author") if isinstance(src, dict) else getattr(src, "author", None)
            permalink = q.get("permalink") or (src.get("permalink") if isinstance(src, dict) else getattr(src, "permalink", ""))
            upvotes = src.get("upvotes") if isinstance(src, dict) else getattr(src, "upvotes", getattr(src, "score", 0))
        else:
            author = "u/unknown"
            permalink = q.get("permalink", "")
            upvotes = 0
            
        quotes_out.append({
            "quote": q.get("quote", ""),
            "author": author or "u/unknown",
            "permalink": permalink or "",
            "upvotes": upvotes or 0
        })
    
    return {
        "query": query,
        "pros": reduce_json.get("pros", [])[:8],
        "cons": reduce_json.get("cons", [])[:8],
        "aspects": sorted(reduce_json.get("aspects", []), key=lambda a: -a.get("count", 0))[:8],
        "quotes": quotes_out,
        "coverage": list({*reduce_json.get("coverage_ids", [])}),
    }


def export_to_json(data: Dict[str, Any], filename: str) -> None:
    """Export data to JSON file."""
    import datetime
    
    # Add timestamp
    data["metadata"]["export_timestamp"] = datetime.datetime.now().isoformat()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
