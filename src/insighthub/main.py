"""Main entry point for InsightHub."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from insighthub.cli import main as cli_main
from insighthub.ui import run_streamlit_app

def main():
    """Main entry point - delegates to CLI or UI based on arguments."""
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        # Run Streamlit UI
        run_streamlit_app()
    else:
        # Run CLI
        cli_main()

if __name__ == "__main__":
    main()
