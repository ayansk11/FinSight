#!/usr/bin/env python3
"""Download a SEC filing and generate a PageIndex tree via Cloud API.

Usage:
    python scripts/download_and_generate.py
    python scripts/download_and_generate.py --ticker MSFT --filing-type 10-K
    python scripts/download_and_generate.py --pdf data/filings/existing.pdf --name AAPL_10K_2024
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mac" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_generate")


def convert_html_to_pdf(html_path: Path) -> Path | None:
    """Convert an HTML filing to PDF using weasyprint."""
    pdf_path = html_path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"  PDF already exists: {pdf_path}")
        return pdf_path

    try:
        from weasyprint import HTML
        HTML(str(html_path)).write_pdf(str(pdf_path))
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  Converted: {pdf_path} ({size_mb:.1f} MB)")
        return pdf_path
    except ImportError:
        print("  weasyprint not installed. Install with: uv pip install weasyprint")
        print("  Also needs system deps: brew install pango")
        return None
    except Exception as e:
        print(f"  Conversion error: {e}")
        return None


async def download_filing(ticker: str, filing_type: str) -> Path | None:
    """Download the most recent filing from SEC EDGAR."""
    from finsight_mac.mcp.sec_edgar import EdgarClient

    client = EdgarClient()
    try:
        print(f"\n[1/2] Searching for {ticker} {filing_type} on SEC EDGAR...")
        filings = await client.search_filings(ticker, filing_type, limit=1)

        if not filings:
            print(f"  No {filing_type} filings found for {ticker}")
            return None

        filing = filings[0]
        company = filing.get("company_name", ticker)
        date = filing.get("filing_date", "unknown")
        doc = filing.get("primary_document", "")
        print(f"  Found: {company} {filing_type} filed {date}")
        print(f"  Document: {doc}")

        print("\n[2/2] Downloading filing...")
        pdf_path = await client.download_filing(filing)

        if pdf_path:
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {pdf_path} ({size_mb:.1f} MB)")
        else:
            print("  Download failed")

        return pdf_path
    finally:
        await client.close()


def generate_tree(pdf_path: Path, doc_name: str | None = None) -> None:
    """Generate a PageIndex tree from a PDF via Cloud API."""
    from finsight_mac.document.pipeline import DocumentPipeline

    pipeline = DocumentPipeline()

    print(f"\nGenerating PageIndex tree for: {pdf_path.name}")
    print("Using PageIndex Cloud API (free tier)")

    tree = pipeline.generate_tree_cloud(
        pdf_path=pdf_path,
        doc_name=doc_name,
        poll_interval=5.0,
        max_wait=600.0,
        add_node_text=True,
    )

    print("\nTree generated successfully!")
    print(f"  Document: {tree.doc_name}")
    print(f"  Description: {tree.doc_description or 'None'}")
    print(f"  Total nodes: {tree.total_nodes}")
    print(f"  Leaf nodes: {len(tree.leaf_nodes)}")
    print("\nTree outline:")
    print(tree.get_tree_outline(max_depth=3))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SEC filing and generate PageIndex tree.",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Company ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--filing-type",
        default="10-K",
        help="SEC filing type (default: 10-K)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Use an existing PDF instead of downloading",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Document name (defaults to ticker_filingtype_date)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only generate tree from existing PDF",
    )

    args = parser.parse_args()

    # Verify API key is set
    from finsight_mac.config import get_settings
    settings = get_settings()

    if not settings.pageindex_api_key:
        print("Error: PAGEINDEX_API_KEY is not set.")
        print("Set it in .env or as an environment variable.")
        print("Get a free key at https://pageindex.ai")
        sys.exit(1)

    print(f"PageIndex API key: {settings.pageindex_api_key[:8]}...")

    if args.pdf:
        # Use existing PDF
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            sys.exit(1)
        generate_tree(pdf_path, args.name)
    else:
        # Download from EDGAR, then generate
        pdf_path = asyncio.run(
            download_filing(args.ticker, args.filing_type)
        )
        if not pdf_path:
            sys.exit(1)

        # SEC EDGAR filings are typically HTML — convert to PDF
        with open(pdf_path, "rb") as f:
            header = f.read(5)

        if header == b"%PDF-":
            generate_tree(pdf_path, args.name)
        else:
            print("\nFiling is HTML format — converting to PDF...")
            converted = convert_html_to_pdf(pdf_path)
            if converted:
                generate_tree(converted, args.name)
            else:
                print("HTML→PDF conversion failed.")
                print(f"HTML saved at: {pdf_path}")
                sys.exit(1)


if __name__ == "__main__":
    main()
