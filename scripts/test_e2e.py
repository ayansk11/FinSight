#!/usr/bin/env python3
"""End-to-end test: query → router → agents → Ollama → response.

Usage:
    python scripts/test_e2e.py
    python scripts/test_e2e.py --query "What are Apple's main risk factors?"
    python scripts/test_e2e.py --model qwen3:8b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mac" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("e2e_test")


async def run_test(query: str, model: str, tree_path: str) -> None:
    """Run the full FinSight pipeline end-to-end."""
    # Override model if specified
    import finsight_mac.config as cfg
    from finsight_mac.config import Settings, get_settings
    from finsight_mac.graph.workflow import get_workflow
    from finsight_mac.llm.ollama_client import OllamaClient

    cfg._settings = Settings(ollama_model=model)
    settings = get_settings()

    print(f"\n{'='*60}")
    print("FinSight End-to-End Test")
    print(f"{'='*60}")
    print(f"Model:  {settings.ollama_model}")
    print(f"Query:  {query}")
    print(f"Tree:   {tree_path}")
    print(f"{'='*60}\n")

    # Step 1: Check Ollama health
    print("[1/5] Checking Ollama connection...")
    client = OllamaClient()
    healthy = await client.check_health()
    if not healthy:
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        return
    print(f"  OK — Connected to {settings.ollama_base_url}")

    # Step 2: Load tree
    print("[2/5] Loading PageIndex tree...")
    with open(tree_path) as f:
        tree_data = json.load(f)
    print(f"  OK — {tree_data['doc_name']}: {len(tree_data['structure'])} top-level nodes")

    # Step 3: Build initial state
    print("[3/5] Building workflow state...")
    initial_state = {
        "query": query,
        "ticker": "AAPL",
        "filings": [{"display_name": tree_data["doc_name"], "pdf_path": None}],
        "trees": [tree_data],
        "route": "",
        "agents_to_run": [],
        "findings": [],
        "risk_scores": [],
        "agent_summaries": [],
        "needs_rlm": False,
        "page_texts_by_filing": {},
        "rlm_result": {},
        "report": "",
        "executive_summary": "",
        "model_name": settings.ollama_model,
        "total_duration_seconds": 0.0,
        "error": None,
    }

    # Step 4: Run workflow
    print("[4/5] Running LangGraph workflow...")
    workflow = get_workflow()
    start = time.time()

    try:
        result = await workflow.ainvoke(initial_state)
    except Exception as e:
        print(f"\nERROR: Workflow failed: {e}")
        import traceback

        traceback.print_exc()
        return

    elapsed = time.time() - start
    print(f"  OK — Completed in {elapsed:.1f}s")

    # Step 5: Display results
    print("\n[5/5] Results")
    print(f"{'='*60}")

    route = result.get("route", "unknown")
    agents = result.get("agents_to_run", [])
    findings = result.get("findings", [])
    risk_scores = result.get("risk_scores", [])
    summaries = result.get("agent_summaries", [])
    report = result.get("report", "")
    exec_summary = result.get("executive_summary", "")

    print(f"\nRoute: {route}")
    print(f"Agents run: {agents}")
    print(f"Total findings: {len(findings)}")
    print(f"Risk scores: {len(risk_scores)}")

    print("\n--- Agent Summaries ---")
    for s in summaries:
        agent = s.get("agent", "?")
        summary = s.get("summary", "")
        duration = s.get("duration", 0)
        print(f"  [{agent}] ({duration:.1f}s) {summary}")

    if exec_summary:
        print("\n--- Executive Summary ---")
        print(exec_summary[:500])

    if report:
        print("\n--- Report (first 1000 chars) ---")
        print(report[:1000])

    if findings:
        print("\n--- Sample Findings ---")
        for i, f in enumerate(findings[:3]):
            content = f.get("content", "")[:200]
            agent = f.get("agent", "?")
            confidence = f.get("confidence", "?")
            print(f"\n  [{i+1}] ({agent}, {confidence})")
            print(f"      {content}")

    if risk_scores:
        print("\n--- Risk Scores ---")
        for rs in risk_scores[:5]:
            tech = rs.get("technique_name", "?")
            severity = rs.get("severity", 0)
            print(f"  {tech}: {severity:.2f}")

    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight end-to-end test")
    parser.add_argument(
        "--query",
        default="What are Apple's main risk factors and how might they impact future revenue?",
        help="Query to test",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="Ollama model name",
    )
    parser.add_argument(
        "--tree",
        default="data/trees/AAPL_10K_2024_structure.json",
        help="Path to PageIndex tree JSON",
    )
    args = parser.parse_args()

    tree_path = Path(args.tree)
    if not tree_path.exists():
        print(f"Tree file not found: {tree_path}")
        sys.exit(1)

    asyncio.run(run_test(args.query, args.model, str(tree_path)))


if __name__ == "__main__":
    main()
