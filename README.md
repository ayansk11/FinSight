# FinSight

**Agentic Financial Document Intelligence & Risk Analyst**

FinSight is a multi-agent AI system that analyzes SEC filings (10-K, 10-Q, 8-K) using a LangGraph-orchestrated pipeline of specialized agents. It combines document intelligence, quantitative analysis, risk classification, and cross-document reasoning to generate investment-grade research reports — all powered by local Ollama inference with no cloud LLM dependencies.

---

## Features

- **Multi-Agent LangGraph Workflow** — Four specialized agents (Document Intelligence, Quantitative Analysis, Risk Classification, Synthesis) orchestrated via LangGraph StateGraph with conditional routing
- **PageIndex Tree Navigation** — Hierarchical document structure maps for precise section-level extraction with citations
- **Docling Document AI** — IBM's AI-powered document conversion with layout analysis, table extraction, and structured Markdown output as the default PDF processor
- **Local Ollama Inference** — Runs entirely offline using Qwen3.5 via Ollama — no API keys required for core analysis. Supports multiple models (Qwen3.5:9b, Qwen3:8b, Qwen3.5:4b, or any Ollama-compatible model)
- **Market Data Integration** — Real-time financial data from Yahoo Finance, FRED, and Finnhub via MCP-style data servers
- **MITRE F3 Risk Framework** — Risk classification using a financial threat taxonomy inspired by MITRE ATT&CK
- **Cross-Document Reasoning (RLM)** — Recursive Language Model integration for multi-filing comparative analysis. When multiple filings are loaded, the router activates RLM synthesis for cross-document reasoning with iterative refinement (up to 15 iterations)
- **Local Tree Generation** — Heuristic heading detection + Ollama refinement for generating document structure trees from raw PDFs
- **Streamlit Web UI** — Interactive chat interface, document manager, analysis dashboard with findings/risk scores/export
- **iPhone Export** — `.finsight` bundle export for companion iOS app

---

## Architecture

```
                              FinSight Architecture
    ┌─────────────────────────────────────────────────────────────────┐
    │                       Streamlit Web UI                          │
    │   ┌──────────┐   ┌──────────────┐   ┌───────────────────┐      │
    │   │   Chat   │   │  Documents   │   │  Analysis Board   │      │
    │   └────┬─────┘   └──────┬───────┘   └─────────┬─────────┘      │
    └────────┼────────────────┼─────────────────────┼─────────────────┘
             │                │                     │
             ▼                ▼                     ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    LangGraph StateGraph                         │
    │                                                                 │
    │   ┌──────────┐    ┌────────────┐    ┌──────────┐               │
    │   │  Router  │───>│  Doc Intel  │───>│  Quant   │ (conditional) │
    │   │  Node    │    │   Agent     │    │  Agent   │               │
    │   └──────────┘    └────────────┘    └────┬─────┘               │
    │                                          │                      │
    │                                          ▼                      │
    │                                    ┌──────────┐                 │
    │                                    │   Risk   │ (conditional)   │
    │                                    │  Agent   │                 │
    │                                    └────┬─────┘                 │
    │                                         │                       │
    │                                    ┌────┴────┐                  │
    │                                    ▼         ▼                  │
    │                            ┌───────────┐ ┌────────────┐        │
    │                            │ Synthesis │ │    RLM     │        │
    │                            │   Agent   │ │ Synthesis  │        │
    │                            └───────────┘ └────────────┘        │
    └─────────────────────────────────────────────────────────────────┘
             │                │                     │
             ▼                ▼                     ▼
    ┌────────────────┐ ┌─────────────┐ ┌─────────────────────────────┐
    │  Ollama LLM    │ │  PageIndex  │ │     MCP Data Servers        │
    │  (Qwen3.5:9b)  │ │   Trees     │ │  ┌───────┐┌───────┐┌─────┐│
    │  localhost:11434│ │  (JSON)     │ │  │ FRED  ││Finnhub││ yFin││
    └────────────────┘ └─────────────┘ │  └───────┘└───────┘└─────┘│
                                        └─────────────────────────────┘
```

---

## Data Flow

```
                              Query Processing Flow

    User Query: "What are Apple's main risk factors?"
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. ROUTER                                                    │
    │    classify_query(query, num_filings)                        │
    │    ├── Detects: risk signals → route = "risk_focused"        │
    │    ├── agents_to_run = ["doc_intel", "risk", "synthesis"]    │
    │    └── needs_rlm = False (single document)                   │
    └──────────────────────┬──────────────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 2. DOC INTEL AGENT                                           │
    │    For each loaded filing:                                   │
    │    ├── Present PageIndex tree outline to LLM                 │
    │    ├── LLM selects relevant node IDs (e.g., Risk Factors)    │
    │    ├── Extract page text for selected sections               │
    │    ├── Analyze each section with LLM                         │
    │    └── Output: Findings[] with precise Citations             │
    │         (filing, section, page range, excerpt)               │
    └──────────────────────┬──────────────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 3. RISK AGENT (conditional — risk_focused route)             │
    │    ├── Classify findings using MITRE F3 taxonomy             │
    │    │   ├── Disclosure Risk (DR-T001..T006)                   │
    │    │   ├── Financial Health (FH-T001..T008)                  │
    │    │   ├── Governance (GV-T001..T005)                        │
    │    │   └── Market Risk (MR-T001..T004)                       │
    │    └── Output: RiskScore[] with severity (0.0-1.0)           │
    └──────────────────────┬──────────────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 4. SYNTHESIS AGENT                                           │
    │    ├── Aggregate all upstream findings + risk scores          │
    │    ├── If RLM: incorporate cross-document reasoning           │
    │    ├── Generate investment-grade report (markdown)            │
    │    └── Generate executive summary (one paragraph)             │
    └──────────────────────┬──────────────────────────────────────┘
                           ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 5. OUTPUT                                                    │
    │    ├── report: Full synthesis report                          │
    │    ├── executive_summary: One-paragraph summary               │
    │    ├── findings: [{content, agent, confidence, citations}]    │
    │    ├── risk_scores: [{tactic, technique, severity, evidence}] │
    │    └── agent_summaries: [{agent, summary, duration}]          │
    └─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
FinSight/
├── shared/                          # Shared core library (finsight_core)
│   └── src/finsight_core/
│       ├── models/
│       │   ├── state.py             # FinSightState — LangGraph state schema
│       │   ├── analysis.py          # Finding, RiskScore, Citation, AgentOutput
│       │   ├── document.py          # Filing, PageIndexTree, TreeNode
│       │   └── export.py            # ExportBundle for iPhone
│       ├── prompts/                 # All LLM prompt templates
│       ├── pageindex/               # PageIndex tree parser & navigator
│       ├── taxonomy/                # MITRE F3 risk taxonomy
│       └── finance/                 # Financial ratio calculations
│
├── mac/                             # Mac application (finsight_mac)
│   └── src/finsight_mac/
│       ├── agents/                  # Four specialized agents
│       │   ├── doc_intel.py         # Document Intelligence Agent
│       │   ├── quant.py             # Quantitative Analysis Agent
│       │   ├── risk.py              # Risk Classification Agent
│       │   └── synthesis.py         # Synthesis & Report Agent
│       ├── graph/
│       │   ├── workflow.py          # LangGraph StateGraph definition
│       │   └── router.py            # Query classification & routing
│       ├── document/
│       │   ├── pipeline.py          # PDF → PageIndex tree pipeline
│       │   └── local_tree_generator.py  # Offline tree generation
│       ├── mcp/                     # Market data servers
│       │   ├── market_data.py       # Aggregator (FRED + Finnhub + yfinance)
│       │   ├── fred.py              # Federal Reserve data
│       │   ├── finnhub_client.py    # Market quotes & fundamentals
│       │   └── yfinance_client.py   # Yahoo Finance (no API key)
│       ├── llm/
│       │   ├── ollama_client.py     # Async Ollama wrapper
│       │   └── rlm_client.py        # Recursive Language Model client
│       ├── ui/                      # Streamlit web interface
│       │   ├── app.py               # Main entry point
│       │   ├── pages/
│       │   │   ├── chat.py          # Chat Q&A interface
│       │   │   ├── documents.py     # PDF/tree management
│       │   │   └── analysis.py      # Analysis dashboard
│       │   └── components/
│       │       ├── tree_viewer.py   # PageIndex tree renderer
│       │       ├── findings_viewer.py
│       │       └── risk_viewer.py
│       └── config.py                # Settings (Pydantic BaseSettings)
│
├── tests/
│   ├── shared/                      # Core library unit tests
│   ├── mac/                         # Agent & workflow unit tests
│   └── e2e/                         # End-to-end tests (requires Ollama)
│
├── scripts/                         # CLI utilities
│   ├── setup_sample_data.py         # One-command sample data setup
│   ├── test_e2e.py                  # Manual E2E test runner
│   ├── generate_tree.py             # Tree generation CLI
│   └── export_for_iphone.py         # .finsight bundle exporter
│
├── data/
│   ├── filings/                     # SEC filing PDFs (gitignored)
│   └── trees/                       # Generated PageIndex trees (gitignored)
│
└── references/                      # Design specs & documentation
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) with `qwen3.5:9b` model
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repo
git clone https://github.com/ayansk11/FinSight.git
cd FinSight

# Install dependencies with uv
uv sync

# Pull the Ollama model
ollama pull qwen3.5:9b

# Copy environment config
cp .env.example .env
# Edit .env to add optional API keys (FRED, Finnhub, PageIndex)
```

### Sample Data Setup

```bash
# Download a sample 10-K from SEC EDGAR and generate a PageIndex tree
# (requires Ollama running with a model pulled)
uv run python scripts/setup_sample_data.py

# Or for a different company:
uv run python scripts/setup_sample_data.py --ticker MSFT
```

Supported tickers: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, V, JNJ.

### Running the UI

```bash
# Start Ollama (if not already running)
ollama serve

# Launch Streamlit
uv run streamlit run mac/src/finsight_mac/ui/app.py
```

### Running Tests

```bash
# Unit tests (no Ollama required)
uv run python -m pytest tests/ -q -m "not e2e"

# E2E tests (requires Ollama + qwen3.5:9b)
uv run python -m pytest tests/e2e/ -v

# Lint
uv run ruff check .
```

### Quick CLI Test

```bash
# Run the full pipeline from command line
uv run python scripts/test_e2e.py --query "What are Apple's main risk factors?"
```

---

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_MODEL` | No | Ollama model name (default: `qwen3.5:9b`) |
| `OLLAMA_BASE_URL` | No | Ollama endpoint (default: `http://localhost:11434`) |
| `FRED_API_KEY` | No | FRED macroeconomic data (free at fred.stlouisfed.org) |
| `FINNHUB_API_KEY` | No | Finnhub market data (free at finnhub.io) |
| `PAGEINDEX_API_KEY` | No | PageIndex cloud tree generation |
| `OPENAI_API_KEY` | No | OpenAI for cloud tree generation fallback |

Core analysis works with **zero API keys** — only Ollama is required. Market data APIs are optional and enhance the Quantitative Agent.

---

## How It Works

### PageIndex Trees

FinSight uses hierarchical **PageIndex trees** to navigate SEC filings efficiently. Instead of processing entire 100+ page documents, the system:

1. Generates a structural tree (PART I → Item 1 → Business → ...) with page ranges
2. Presents the tree outline to the LLM for intelligent section selection
3. Extracts only relevant pages, maintaining precise citations

Trees can be generated via:
- **Local Ollama** — Heuristic heading detection + LLM refinement (offline, no API key)
- **PageIndex Cloud API** — High-quality cloud-based generation
- **Pre-generated** — Load from `data/trees/` directory

### Agent Routing

The Router analyzes query intent and selects the appropriate agent combination:

| Route | Agents | Trigger |
|---|---|---|
| `single_doc` | doc_intel → synthesis | Default for single-filing queries |
| `quantitative` | doc_intel → quant → synthesis | Revenue, ratios, margins, P/E |
| `risk_focused` | doc_intel → risk → synthesis | Risk factors, governance, fraud |
| `multi_doc` | doc_intel → quant/risk → RLM synthesis | Cross-filing comparison queries |

### Supported LLM Models

FinSight uses Ollama for local inference and supports any Ollama-compatible model. Configure via the `OLLAMA_MODEL` environment variable:

| Model | Ollama ID | Size | Notes |
|---|---|---|---|
| Qwen3.5-9B | `qwen3.5:9b` | ~6GB | Default — best balance of quality and speed |
| Qwen3-8B (RLM) | `qwen3:8b-q4_K_M` | ~5GB | Optimized for cross-document reasoning |
| Qwen3.5-4B | `qwen3.5:4b` | ~3GB | Lightweight, good for quick queries |
| Qwen3.5-2B | `qwen3.5:2b` | ~1.5GB | Minimal memory footprint |
| Qwen3.5-0.8B | `qwen3.5:0.8b` | ~0.5GB | Ultra-light, fastest inference |

Select your model from the dropdown in the Streamlit sidebar. The default (`OLLAMA_MODEL` env var) can be overridden at runtime.

### Recursive Language Model (RLM)

When multiple filings are loaded (e.g., comparing AAPL 10-K 2023 vs 2024), the Router automatically activates **RLM synthesis** instead of standard synthesis. RLM performs iterative cross-document reasoning:

1. Generates initial comparative analysis from all upstream findings
2. Iteratively refines the analysis (up to `MAX_RLM_ITERATIONS`, default 15)
3. Produces a unified cross-filing report with comparative insights

RLM is triggered automatically — no configuration needed beyond loading multiple documents.

### Docling Document AI

FinSight uses [Docling](https://github.com/DS4SD/docling) (IBM) as its default PDF processor for:

- **Layout analysis** — Correct reading order for multi-column layouts
- **Table extraction** — Structured table data from financial statements
- **Page-level Markdown** — Clean formatted text preserving document structure

Docling results are cached to disk for fast reprocessing. Falls back to PyMuPDF if Docling processing fails on a specific document.

### Risk Classification

The Risk Agent uses a financial threat taxonomy inspired by MITRE ATT&CK:

- **Disclosure Risk** — Material omissions, vague language, delayed reporting
- **Financial Health** — Revenue concentration, debt levels, cash flow issues
- **Governance** — Related party transactions, executive compensation, board independence
- **Market Risk** — Concentration risk, regulatory exposure, competitive threats

Each risk receives a severity score (0.0–1.0) with supporting evidence and citations.

---

## Test Coverage

- **129 unit tests** — Models, agents, workflow routing, tree generation, MCP clients
- **10 E2E tests** — Full pipeline, live market data, real PDF processing
- All tests auto-skip when Ollama/network unavailable

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM | Ollama (Qwen3.5:9b default, multi-model) |
| Document AI | Docling (IBM) — layout analysis, tables |
| Document Parsing | PyMuPDF, PageIndex |
| Cross-Doc Reasoning | RLM (Recursive Language Model) |
| Market Data | yfinance, FRED API, Finnhub |
| Data Models | Pydantic v2 |
| Web UI | Streamlit |
| Package Manager | uv (workspace) |
| Testing | pytest, pytest-asyncio |
| Linting | Ruff |

---

## License

MIT
