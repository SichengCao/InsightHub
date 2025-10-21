# InsightHub 2.0 🚀

**AI-powered cross-platform review analysis with Reddit + YouTube integration**

InsightHub is an intelligent review analysis platform that uses GPT to analyze reviews from multiple platforms (Reddit + YouTube) and provide comprehensive insights across any domain - from tech products to local services.

## ✨ Features

### 🧠 GPT-Only Pipeline
- **Intent Detection**: Automatically detects RANKING, SOLUTION, or GENERIC query intents
- **Dynamic Aspect Generation**: Context-aware aspects for any domain (tech, services, locations)
- **Entity Extraction & Ranking**: Extracts and ranks entities with confidence scores
- **Per-comment Analysis**: Individual comment scoring with aspect breakdowns
- **Solution Clustering**: Groups similar solutions for problem-solving queries

### 🌐 Cross-Platform Analysis
- **Multi-Platform Data**: Collects reviews from Reddit and YouTube
- **Dynamic Platform Weighting**: Intelligent weight allocation based on query domain and data quality
- **Unified Analysis**: Combines insights from multiple sources for comprehensive results
- **Platform-Specific Optimization**: Reddit uses LLM search planning, YouTube uses relevance filtering

### 🔍 Universal Search
- **Reddit**: LLM-powered search planning with intelligent term/subreddit discovery
- **YouTube**: Relevance-based video and comment filtering
- **Quality Filtering**: Advanced comment filtering and deduplication across platforms
- **Flexible Search**: Works for any query (products, locations, services)

### 📊 Analysis Capabilities
- **Comprehensive Summaries**: GPT-generated pros/cons with evidence
- **Entity Rankings**: Ranked lists with scores, mentions, and quotes
- **Solution Clusters**: Grouped solutions with steps and caveats
- **Aspect Analysis**: Domain-specific aspect scoring

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SichengCao/InsightHub.git
cd InsightHub

# Install dependencies
pip install -e .

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export REDDIT_CLIENT_ID="your-reddit-client-id"
export REDDIT_CLIENT_SECRET="your-reddit-client-secret"
export YOUTUBE_API_KEY="your-youtube-api-key"  # Optional
```

### Usage

#### Command Line Interface

```bash
# Analyze reviews
insighthub analyze "iPhone 15" --limit 20 --out results.json

# Scrape only
insighthub scrape "Tesla Model Y" --limit 10

# Export data
insighthub export results.json --pretty
```

#### Web Interface

```bash
# Launch Streamlit UI
insighthub ui
# or
streamlit run src/insighthub/ui/streamlit_app.py
```

**Cross-Platform Mode**: Enable in the UI to analyze reviews from both Reddit and YouTube simultaneously.

## 📁 Project Structure

```
InsightHub/
├── src/insighthub/
│   ├── core/                    # Core modules
│   │   ├── models.py            # Data models
│   │   ├── config.py            # Configuration
│   │   ├── aspect.py            # Aspect detection
│   │   ├── scoring.py           # Scoring algorithms
│   │   └── cross_platform_models.py # Cross-platform data models
│   ├── services/                # External services
│   │   ├── llm.py               # GPT integration
│   │   ├── reddit_client.py     # Reddit API
│   │   ├── youtube_client.py    # YouTube API
│   │   └── cross_platform_manager.py # Multi-platform orchestration
│   ├── ui/                      # User interfaces
│   │   └── streamlit_app.py
│   ├── utils/                   # Utilities
│   │   └── data_prep.py
│   ├── cli.py                   # Command line interface
│   └── main.py                  # Entry point
├── config/                      # Configuration files
│   ├── aspects/                 # Domain-specific aspects
│   ├── weights.yaml            # Scoring weights
│   └── platform_weights.yaml   # Cross-platform weighting
├── tests/                       # Test suite
├── docs/                        # Documentation
└── examples/                    # Usage examples
```

## 🎯 Query Types

### RANKING Queries
Compare and rank specific items:
- `"iPhone vs Samsung Galaxy S24"`
- `"best golf course in bay area"`
- `"Tesla Model Y vs BMW iX"`

**Output**: Ranked entities with scores, mentions, confidence, and quotes

### SOLUTION Queries
Find solutions to problems:
- `"iPhone battery drain fix"`
- `"Tesla Model Y wind noise solution"`
- `"MacBook Pro overheating fix"`

**Output**: Solution clusters with steps, caveats, and evidence

### GENERIC Queries
General discussion and reviews:
- `"iPhone 15"`
- `"Tesla Model Y"`
- `"Nintendo Switch"`

**Output**: Overall rating, aspect scores, pros/cons, and representative quotes

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Reddit API (optional, uses mock data if not provided)
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret
REDDIT_USER_AGENT=InsightHub/2.0

# YouTube API (optional, uses mock data if not provided)
YOUTUBE_API_KEY=your-youtube-api-key

# Optional
CACHE_DIR=.cache
LOG_LEVEL=INFO
```

### Configuration Files

- `config/aspects/tech_products.yaml` - Tech product aspects
- `config/weights.yaml` - Scoring weights
- `config/platform_weights.yaml` - Cross-platform weighting strategy

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Test specific functionality
python -m pytest tests/test_scoring.py -v
```

## 📈 Examples

### CLI Examples

```bash
# Analyze iPhone reviews
insighthub analyze "iPhone 15" --limit 20

# Compare products
insighthub analyze "iPhone vs Samsung Galaxy S24" --limit 15

# Find solutions
insighthub analyze "iPhone battery drain fix" --limit 10

# Export results
insighthub export analysis_results.json --pretty
```

### Python API

```python
from insighthub import RedditService, LLMServiceFactory
from insighthub.services.cross_platform_manager import CrossPlatformManager
from insighthub.core.scoring import aggregate_generic, rank_entities

# Single platform analysis
reddit_service = RedditService()
llm_service = LLMServiceFactory.create()

reviews = reddit_service.scrape("iPhone 15", limit=20)
intent_schema = llm_service.detect_intent_and_schema("iPhone 15")
annotations = llm_service.annotate_comments_with_gpt(reviews, intent_schema.aspects)

# Cross-platform analysis
cross_platform_manager = CrossPlatformManager()
results = cross_platform_manager.search_cross_platform(
    query="iPhone 15",
    platforms=[Platform.REDDIT, Platform.YOUTUBE],
    limit_per_platform=20
)

# Process results
if intent_schema.intent == "RANKING":
    ranking = rank_entities(annotations, upvote_map, intent_schema.entity_type)
    print(f"Top entities: {[item.name for item in ranking[:5]]}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Key Features

### Cross-Platform Intelligence
- **Dynamic Weighting**: Automatically adjusts platform weights based on query domain and data quality
- **Unified Analysis**: Combines Reddit discussions with YouTube reviews for comprehensive insights
- **Platform Optimization**: Reddit uses LLM search planning, YouTube uses relevance filtering

### Performance Optimized
- **Fast Filtering**: Optimized comment filtering (231s → 0.0s improvement)
- **Parallel Processing**: Simultaneous data collection from multiple platforms
- **Smart Caching**: LLM response caching for faster repeated queries

## 🙏 Acknowledgments

- OpenAI for GPT API
- Reddit for PRAW library
- YouTube for Data API v3
- Streamlit for the web interface
- The open-source community for inspiration

---

**InsightHub 2.0** - Cross-platform review analysis powered by GPT 🚀