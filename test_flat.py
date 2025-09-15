#!/usr/bin/env python3
"""Test script for flat module structure."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from config import settings, ensure_config_files
        print("✅ config imported")
        
        from models import Review, AnalysisSummary, AspectScore
        print("✅ models imported")
        
        from reddit_client import RedditService
        print("✅ reddit_client imported")
        
        from llm import LLMServiceFactory
        print("✅ llm imported")
        
        from sentiment import VADERSentimentAnalyzer
        print("✅ sentiment imported")
        
        from aspect import YAMLAspectDetector
        print("✅ aspect imported")
        
        from scoring import create_analysis_summary, compute_aspect_scores
        print("✅ scoring imported")
        
        from data_prep import prepare_export, export_to_json
        print("✅ data_prep imported")
        
        from cli import main
        print("✅ cli imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_services():
    """Test that services can be initialized."""
    try:
        from reddit_client import RedditService
        from llm import LLMServiceFactory
        from sentiment import VADERSentimentAnalyzer
        from aspect import YAMLAspectDetector
        
        reddit = RedditService()
        print("✅ RedditService initialized")
        
        llm = LLMServiceFactory.create()
        print("✅ LLMService initialized")
        
        sentiment = VADERSentimentAnalyzer()
        print("✅ SentimentAnalyzer initialized")
        
        aspect = YAMLAspectDetector()
        print("✅ AspectDetector initialized")
        
        return True
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without OpenAI keys."""
    try:
        from llm import LLMServiceFactory
        from sentiment import VADERSentimentAnalyzer
        from aspect import YAMLAspectDetector
        
        # Test sentiment analysis
        sentiment = VADERSentimentAnalyzer()
        result = sentiment.analyze("This is a great product!")
        print(f"✅ Sentiment analysis: {result}")
        
        # Test aspect detection
        aspect = YAMLAspectDetector()
        aspects = aspect.detect_aspects("The battery life is amazing and the camera quality is excellent")
        print(f"✅ Aspect detection: {aspects}")
        
        # Test LLM fallback
        llm = LLMServiceFactory.create()
        plan = llm.plan_reddit_search("iPhone 15")
        print(f"✅ LLM search planning: {plan}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing flat module structure...")
    print()
    
    print("1. Testing imports...")
    imports_ok = test_imports()
    print()
    
    print("2. Testing service initialization...")
    services_ok = test_services()
    print()
    
    print("3. Testing basic functionality...")
    functionality_ok = test_basic_functionality()
    print()
    
    if imports_ok and services_ok and functionality_ok:
        print("🎉 All tests passed! Flat module structure is working.")
    else:
        print("❌ Some tests failed. Check the errors above.")
        sys.exit(1)
