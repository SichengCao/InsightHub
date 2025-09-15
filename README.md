# InsightHub

Cross-platform review intelligence platform for analyzing Reddit reviews with AI-powered sentiment and aspect scoring.

## Features

- **Reddit Review Scraping**: Extract reviews from Reddit using PRAW
- **Sentiment Analysis**: VADER-based sentiment analysis with 1-5 star ratings
- **Aspect Detection**: YAML-driven aspect classification for tech products
- **AI-Powered Analysis**: OpenAI integration for pros/cons generation
- **Web UI**: Clean Streamlit interface for interactive analysis
- **CLI Tools**: Command-line interface for batch processing

## Quick Start

1. **Install**: `pip install -e .`
2. **Configure**: Copy `.env.example` to `.env` and add your API keys
3. **Run UI**: `insighthub ui`
4. **CLI Analysis**: `insighthub analyze "iPhone 15" --limit 50 --out results.json`

## Commands

- `insighthub scrape <query>` - Scrape Reddit reviews
- `insighthub analyze <query> --out <file>` - Full analysis pipeline
- `insighthub ui` - Launch web interface

## Configuration

- **Reddit API**: Set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
- **OpenAI API**: Set `OPENAI_API_KEY` (optional, falls back to mock data)
- **Aspects**: Configure in `config/aspects/tech_products.yaml`
- **Weights**: Adjust in `config/weights.yaml`

## Architecture

```
insighthub/
├── analysis/          # Sentiment analysis and aspect detection
├── services/          # Reddit scraping and LLM services
├── ui/               # Streamlit web interface
├── reporting/        # Data export and preparation
├── config/           # YAML configuration files
└── tests/            # Unit tests
```

## Technologies

- **Python 3.8+**
- **Streamlit** - Web UI
- **OpenAI GPT** - AI analysis
- **VADER** - Sentiment analysis
- **PRAW** - Reddit API
- **Pydantic** - Data validation

## Recent Updates

- ✅ **Precise GPT Summaries**: More specific and detailed analysis
- ✅ **Simplified Comment Display**: Top 5 positive/negative comments
- ✅ **Enhanced Mock Data**: 50 comprehensive reviews per analysis
- ✅ **Glassmorphism UI**: Modern, professional interface
- ✅ **Production Ready**: Complete modular architecture