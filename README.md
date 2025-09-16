# InsightHub 2.0 🚀

**AI-powered Reddit review analysis platform with GPT-only pipeline**

InsightHub is an intelligent review analysis platform that uses GPT to analyze Reddit comments and provide comprehensive insights across any domain - from tech products to local services.

## ✨ Features

### 🧠 GPT-Only Pipeline
- **Intent Detection**: Automatically detects RANKING, SOLUTION, or GENERIC query intents
- **Dynamic Aspect Generation**: Context-aware aspects for any domain (tech, services, locations)
- **Entity Extraction & Ranking**: Extracts and ranks entities with confidence scores
- **Per-comment Analysis**: Individual comment scoring with aspect breakdowns
- **Solution Clustering**: Groups similar solutions for problem-solving queries

### 🔍 Universal Reddit Search
- **LLM-powered Search Planning**: Intelligent term/subreddit discovery
- **Quality Filtering**: Advanced comment filtering and deduplication
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

## 📁 Project Structure

```
InsightHub/
├── src/insighthub/
│   ├── core/           # Core modules
│   │   ├── models.py   # Data models
│   │   ├── config.py   # Configuration
│   │   ├── aspect.py   # Aspect detection
│   │   └── scoring.py  # Scoring algorithms
│   ├── services/       # External services
│   │   ├── llm.py      # GPT integration
│   │   └── reddit_client.py # Reddit API
│   ├── ui/             # User interfaces
│   │   └── streamlit_app.py
│   ├── utils/          # Utilities
│   │   └── data_prep.py
│   ├── cli.py          # Command line interface
│   └── main.py         # Entry point
├── config/             # Configuration files
├── tests/              # Test suite
├── docs/               # Documentation
└── examples/           # Usage examples
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

# Optional
CACHE_DIR=.cache
LOG_LEVEL=INFO
```

### Configuration Files

- `config/aspects/tech_products.yaml` - Tech product aspects
- `config/weights.yaml` - Scoring weights

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
from insighthub.core.scoring import aggregate_generic, rank_entities

# Initialize services
reddit_service = RedditService()
llm_service = LLMServiceFactory.create()

# Scrape and analyze
reviews = reddit_service.scrape("iPhone 15", limit=20)
intent_schema = llm_service.detect_intent_and_schema("iPhone 15")
annotations = llm_service.annotate_comments_with_gpt(reviews, intent_schema.aspects)

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

## 🙏 Acknowledgments

- OpenAI for GPT API
- Reddit for PRAW library
- Streamlit for the web interface
- The open-source community for inspiration

---

**InsightHub 2.0** - Intelligent review analysis powered by GPT 🚀