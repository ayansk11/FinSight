#!/usr/bin/env python3
"""Download a sample SEC filing and generate a PageIndex tree.

One-command setup for first-time users:
    uv run python scripts/setup_sample_data.py

This downloads Apple's latest 10-K from SEC EDGAR (free, no API key)
and generates a PageIndex tree using local Ollama.

Usage:
    python scripts/setup_sample_data.py
    python scripts/setup_sample_data.py --ticker MSFT
    python scripts/setup_sample_data.py --skip-tree
    python scripts/setup_sample_data.py --model qwen3.5:4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import urllib.request
from pathlib import Path

# Add project root to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mac" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup")

# CIK numbers for common tickers (SEC EDGAR uses CIK, not ticker)
TICKER_TO_CIK: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "META": "0001326801",
    "NVDA": "0001045810",
    "JPM": "0000019617",
    "V": "0001403161",
    "JNJ": "0000200406",
}

SEC_USER_AGENT = "FinSight research@example.com"


def _sec_request(url: str) -> bytes:
    """Make a request to SEC EDGAR with the required User-Agent header."""
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def find_latest_10k(cik: str) -> tuple[str, str, str]:
    """Find the latest 10-K filing for a company on SEC EDGAR.

    Returns (accession_number, primary_document, filing_date).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.info(f"Querying SEC EDGAR for CIK {cik}...")
    data = json.loads(_sec_request(url))

    filings = data["filings"]["recent"]
    for i in range(len(filings["form"])):
        if filings["form"][i] == "10-K":
            return (
                filings["accessionNumber"][i],
                filings["primaryDocument"][i],
                filings["filingDate"][i],
            )

    raise ValueError(f"No 10-K filing found for CIK {cik}")


def download_filing(
    cik: str,
    accession: str,
    document: str,
    output_path: Path,
) -> None:
    """Download a filing document from SEC EDGAR."""
    acc_no_dashes = accession.replace("-", "")
    cik_num = cik.lstrip("0")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_no_dashes}/{document}"
    logger.info(f"Downloading {document} from SEC EDGAR...")
    logger.info(f"  URL: {url}")

    content = _sec_request(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    size_mb = len(content) / (1024 * 1024)
    logger.info(f"  Saved to {output_path} ({size_mb:.1f} MB)")


def check_ollama(model: str) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        data = json.loads(_sec_request("http://localhost:11434/api/tags").decode())
    except Exception:
        return False

    model_names = [m["name"] for m in data.get("models", [])]
    return any(model in name for name in model_names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a sample SEC filing and generate a PageIndex tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                           # Download AAPL 10-K + generate tree
  %(prog)s --ticker MSFT             # Use Microsoft instead
  %(prog)s --skip-tree               # Only download, skip tree generation
  %(prog)s --model qwen3.5:4b        # Use a smaller Ollama model
""",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Company ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--skip-tree",
        action="store_true",
        help="Skip PageIndex tree generation (only download the filing)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model for tree generation (default: from config)",
    )

    args = parser.parse_args()
    ticker = args.ticker.upper()

    cik = TICKER_TO_CIK.get(ticker)
    if not cik:
        print(f"Error: Unknown ticker '{ticker}'.")
        print(f"Supported tickers: {', '.join(sorted(TICKER_TO_CIK))}")
        print("For other companies, find their CIK at https://www.sec.gov/cgi-bin/browse-edgar")
        sys.exit(1)

    # Set up paths
    from finsight_mac.config import get_settings

    settings = get_settings()
    settings.ensure_dirs()

    print()
    print("=" * 60)
    print("  FinSight — Sample Data Setup")
    print("=" * 60)
    print()

    # Step 1: Find and download the filing
    accession, document, filing_date = find_latest_10k(cik)
    doc_name = f"{ticker}_10K_{filing_date[:4]}"
    ext = Path(document).suffix  # .htm or .pdf
    filing_path = settings.filings_dir / f"{doc_name}{ext}"

    if filing_path.exists():
        logger.info(f"Filing already exists: {filing_path}")
        print(f"  [skip] Filing already downloaded: {filing_path.name}")
    else:
        print(f"  Downloading {ticker} 10-K ({filing_date})...")
        download_filing(cik, accession, document, filing_path)
        print(f"  [done] Saved to {filing_path}")

    print()

    # Step 2: Generate PageIndex tree
    tree_path = settings.trees_dir / f"{doc_name}_structure.json"

    if args.skip_tree:
        print("  [skip] Tree generation skipped (--skip-tree)")
    elif tree_path.exists():
        logger.info(f"Tree already exists: {tree_path}")
        print(f"  [skip] Tree already generated: {tree_path.name}")
    else:
        # Check Ollama
        model = args.model or settings.ollama_model
        if not check_ollama(model):
            print(f"  [warn] Ollama not running or model '{model}' not found.")
            print("         Start Ollama and pull the model:")
            print("           ollama serve")
            print(f"           ollama pull {model}")
            print()
            print("         Then re-run: uv run python scripts/setup_sample_data.py")
            sys.exit(1)

        print(f"  Generating PageIndex tree with {model}...")
        print("  (this may take 2-5 minutes depending on document size)")

        # Check if Groq fallback is available
        from finsight_mac.llm.groq_client import GroqClient

        if GroqClient.is_available():
            print("  Groq fallback: enabled (will auto-switch if Ollama times out)")
        else:
            print("  Groq fallback: not configured (set GROQ_API_KEY for faster processing)")
        print()

        # Override model if specified
        if args.model:
            import finsight_mac.config as cfg

            cfg._settings = type(settings)(ollama_model=args.model)

        from finsight_mac.document.pipeline import DocumentPipeline

        pipeline = DocumentPipeline()
        tree = asyncio.run(
            pipeline.generate_tree_local(
                pdf_path=filing_path,
                doc_name=doc_name,
            )
        )
        print(f"  [done] Tree saved: {tree_path.name}")
        print(f"         {tree.total_nodes} nodes, {len(tree.leaf_nodes)} leaves")

    # Summary
    print()
    print("=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print()
    print("    1. Start Ollama (if not running):")
    print("       ollama serve")
    print()
    print("    2. Launch the Streamlit UI:")
    print("       uv run streamlit run mac/src/finsight_mac/ui/app.py")
    print()
    print("    3. In the UI:")
    print("       - Go to Documents tab → Load the tree")
    print(f'       - Go to Chat tab → Ask: "What are {ticker}\'s main risk factors?"')
    print("       - Go to Analysis tab → View findings and risk scores")
    print()
    print("    Or run from CLI:")
    print(f'       uv run python scripts/test_e2e.py --query "Summarize {ticker}\'s risk factors"')
    print()


if __name__ == "__main__":
    main()
