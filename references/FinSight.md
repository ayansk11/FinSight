# Building an AI career in 2026: skills, strategy, and a standout finance project

**The AI job market in 2026 demands production-grade agentic AI skills, not academic credentials.** Hiring managers at top labs now filter for MCP fluency, LangGraph orchestration experience, and deployed multi-agent systems — skills that barely existed 18 months ago. The single most effective way to demonstrate these skills is an end-to-end project that integrates cutting-edge techniques in a real domain. This report maps every required skill across six AI roles, identifies the gaps in existing open-source finance AI projects, and proposes a novel financial analyst system — **FinSight** — that would serve as a powerful portfolio centerpiece for 2026 hiring.

---

## Part 1: What the 2026 AI job market actually requires

### The universal baseline across all six roles

Every AI/ML role in 2026 shares a non-negotiable foundation. **Python appears in 71–100% of all AI engineer postings.** PyTorch dominates at **46.9%** of postings (TensorFlow at 38.8%), and SQL proficiency is tested in nearly every technical screen. Beyond languages, the 2026 baseline includes Docker/Kubernetes for deployment, at least one cloud platform (AWS leads at 32.9%, Azure at 26%), and Git-based version control.

The knowledge baseline has shifted dramatically. **RAG architectures, agentic AI, and LLM integration are now core requirements, not differentiators.** A 365 Data Science analysis of 903+ postings confirmed that 75% of AI job listings seek domain specialists with deep, focused knowledge rather than generalists. The era of "I know a little ML" is over.

Here is the complete skills matrix by role:

| Skill Area | AI Engineer | Data Scientist | Agentic AI Engineer | LLM Engineer | RL Engineer | ML Engineer (LLM/Agents) |
|---|---|---|---|---|---|---|
| **Core Languages** | Python, Java, SQL | Python, R, SQL | Python, C# | Python, C++, Java | Python, C++ | Python, Java, Scala |
| **Key Frameworks** | PyTorch, LangChain, LangGraph, HuggingFace | scikit-learn, XGBoost, Spark | LangGraph, Semantic Kernel, AutoGen, CrewAI | PyTorch, JAX, vLLM, TRL | PyTorch, JAX, Stable-Baselines3 | PyTorch, LangChain, LangGraph, SparkML |
| **Must-Know Concepts** | RAG, agents, fine-tuning, prompt engineering | Statistics, ML algorithms, feature engineering | Multi-agent orchestration, tool calling, guardrails | RLHF/DPO, tokenization, inference optimization | Policy optimization, reward shaping, sim-to-real | End-to-end ML systems, distributed training, RAG |
| **Deployment** | Docker, K8s, CI/CD, MLflow | Tableau, Airflow, dbt | REST APIs, microservices, async programming | TensorRT-LLM, quantization, model compression | HPC clusters, GPU programming | Docker, K8s, MLflow, Terraform |
| **Education** | BS minimum, MS common | BS minimum (70% require degree) | BS + 3yr GenAI experience | MS common, 3+ yr NLP | **PhD strongly preferred** | BS minimum, PhD for 36.2% |
| **Salary Range** | $143K–$500K+ | $160K–$200K (median) | $150K–$350K+ | $168K–$300K+ equity | $180K–$400K+ | $160K–$470K+ |

### Five emerging skills that separate 2026 candidates from the rest

**MCP (Model Context Protocol)** is the single biggest new signal in 2026 hiring. After Anthropic donated MCP to the Linux Foundation in December 2025, OpenAI, Google, Microsoft, and Amazon adopted it. Over **60,000 projects** now use MCP, and IBM includes it in their RAG and Agentic AI certificate. Multiple job postings (BillGO, enterprise roles) explicitly list MCP proficiency. As one hiring analysis put it: "If you're building agents in 2026 and not using MCP, you're building on legacy infrastructure."

**Context engineering** has replaced prompt engineering as the central design discipline. Anthropic defines it as designing "what data, knowledge, tools, memory, and structure to provide the model at inference time." This is broader than writing good prompts — it encompasses retrieval strategy, memory architecture, and tool selection.

**Agentic RAG** (corrective, self-corrective, adaptive, and graph-based RAG) has become table stakes for AI Engineer and LLM Engineer roles. Single-pass retrieve-and-generate systems are considered outdated. Production systems now use multi-step retrieval with validation, source checking, query reformulation, and fallback to web search.

**GraphRAG**, combining vector search with knowledge graphs, achieves up to **99% retrieval accuracy** in Microsoft Research benchmarks. Hiring managers increasingly expect candidates to understand when vector search alone is insufficient and how knowledge graphs enhance retrieval quality.

**Agent evaluation and observability** — tracing, logging, and dashboards for prompts, responses, and failure cases — has emerged as a critical production skill. Weights & Biases, LangSmith, and custom observability stacks are now expected knowledge for senior roles.

### What actually gets you hired at top labs

Anthropic's hiring process reveals what matters most: **"If you have done interesting independent research, written an insightful blog post, or made substantial contributions to open-source software, put that at the TOP of your resume."** Roughly 50% of Anthropic's technical staff lack PhDs — production skills and demonstrated thinking matter more.

Portfolio projects that work, based on analysis of successful candidates at Anthropic, OpenAI, and Meta:

- A **production RAG application** with semantic search, source citations, evaluation metrics, hybrid search, and a deployed UI
- A **multi-agent system** with 3+ specialized agents, tool integration, handoff mechanisms, and human-in-the-loop patterns
- A **paper reimplementation** in PyTorch from scratch with comprehensive documentation
- **Deployed systems** with Docker, monitoring dashboards, and architecture diagrams

Projects that get immediately rejected: Jupyter notebooks without deployment, MNIST/Titanic tutorial-level work, missing READMEs, no deployment links.

### Intern versus full-time expectations

The gap between intern and full-time requirements is narrower than ever but still significant. Interns need Python fluency, basic PyTorch/TensorFlow, LLM API integration, basic RAG pipelines, and **exposure** to agent frameworks like LangChain. Full-time roles demand **production-level** deployment at scale, full CI/CD and Kubernetes expertise, fine-tuning and RLHF experience, and multi-agent orchestration with production guardrails. Intern compensation at top companies runs **$45–65/hour** (NVIDIA), while only **2.5% of AI engineering roles** target candidates with 0–2 years of experience — making a standout portfolio project essential for breaking in.

---

## Part 2: A novel finance AI project — FinSight

### Why existing projects leave a massive gap

The open-source finance AI landscape is dominated by the AI4Finance Foundation's ecosystem: **FinGPT** (~18.6K stars) for financial sentiment and LLM fine-tuning, **FinRL** (~14K stars) for reinforcement learning trading, **FinRobot** (~6.3K stars) for multi-agent equity research, and **OpenBB** (~37K stars) as a data aggregation platform. These are strong projects, but a systematic gap analysis reveals critical voids.

**No existing open-source project provides explainable, auditable financial document intelligence with guardrails.** FinRobot generates equity research reports but depends entirely on proprietary GPT-4 APIs, has no human-in-the-loop workflow, no guardrails, and no audit trail. Patronus AI found that even GPT-4 achieves only **~56% accuracy** on realistic financial QA from SEC filings. FinGPT handles sentiment but not deep document analysis. FinRL handles trading but has zero NLP capabilities. No project combines document intelligence, risk classification, multi-agent orchestration, and compliance-ready audit trails.

The proposed project — **FinSight** — fills exactly these gaps while integrating every cutting-edge technique that 2026 hiring managers want to see.

### FinSight: Autonomous Financial Document Intelligence & Risk Analyst

**Core concept:** An agentic financial analyst that ingests SEC filings, earnings reports, and macroeconomic data, then performs multi-step reasoning to produce investment-grade analysis with explainable reasoning chains, risk classification using a MITRE ATT&CK-inspired financial taxonomy, and full audit trails — all running locally on a MacBook Air M4.

**What makes it novel:**

1. **PageIndex-powered document intelligence** — not generic RAG, but layout-aware hierarchical tree indexing that achieves **98.7% accuracy on FinanceBench** (versus ~31% for GPT-4o alone and ~45% for Perplexity). PageIndex transforms 200-page 10-K filings into navigable JSON tree structures, preserving table layouts, cross-references ("see Note 12"), and footnote associations that vector RAG destroys.

2. **RLM (Recursive Language Model) for massive financial corpora** — the MIT paper by Zhang, Kraska, and Khattab (December 2025) enables processing documents **2 orders of magnitude beyond** model context windows. The LLM treats the entire corpus as an external environment variable, recursively calling itself via a Python REPL to grep, partition, and analyze specific sections. This means FinSight can reason across an entire company's multi-year filing history without context rot.

3. **MITRE F3-inspired financial risk taxonomy** — MITRE announced the Fight Financial Fraud (F3) framework in May 2025, creating an ATT&CK-style taxonomy for financial fraud. FinSight adapts this into a structured classification system for financial risks detected in documents: mapping language patterns in 10-K risk factors, MD&A sections, and earnings calls to specific fraud/risk tactics and techniques.

4. **Complements FinSent-CoT** — where FinSent-CoT performs chain-of-thought sentiment analysis on financial text, FinSight operates upstream and downstream: it extracts structured intelligence from complex documents (upstream) and classifies systemic risks across filings (downstream). The two projects form a complete pipeline.

### Complete architecture and tech stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    FinSight Architecture                         │
│                                                                 │
│  ┌──────────┐    ┌──────────────────────────────────────────┐  │
│  │  MCP Hub  │◄──►│         LangGraph Orchestrator           │  │
│  │           │    │  ┌────────┐ ┌────────┐ ┌─────────────┐  │  │
│  │ SEC EDGAR │    │  │Document│ │Quant   │ │Risk         │  │  │
│  │ FRED      │    │  │Intel   │ │Analysis│ │Classifier   │  │  │
│  │ Finnhub   │    │  │Agent   │ │Agent   │ │Agent (F3)   │  │  │
│  │ yfinance  │    │  └───┬────┘ └───┬────┘ └──────┬──────┘  │  │
│  │ PageIndex │    │      │          │              │         │  │
│  └──────────┘    │  ┌───▼──────────▼──────────────▼──────┐  │  │
│                  │  │         Shared State Graph           │  │  │
│  ┌──────────┐   │  │  (findings, citations, risk scores)  │  │  │
│  │NeMo      │   │  └───────────────┬─────────────────────┘  │  │
│  │Guardrails│◄──┤                  │                         │  │
│  │+ HITL    │   │  ┌───────────────▼─────────────────────┐  │  │
│  └──────────┘   │  │      Synthesis & Report Agent        │  │  │
│                  │  │  (RLM for cross-document reasoning)  │  │  │
│  ┌──────────┐   │  └───────────────┬─────────────────────┘  │  │
│  │Local LLM │   │                  │                         │  │
│  │Qwen3 8B  │◄──┤                  ▼                         │  │
│  │Q4_K_M    │   │     ┌─────────────────────────┐           │  │
│  │via Ollama│   │     │  Streamlit Dashboard UI  │           │  │
│  └──────────┘   │     └─────────────────────────┘           │  │
│                  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Complete tech stack (all free):**

| Layer | Technology | Cost |
|---|---|---|
| **LLM (local)** | Qwen 3 8B Q4_K_M via Ollama (~5GB, 20-30 tok/s on M4) | Free |
| **Reasoning LLM** | DeepSeek-R1-Distill-Qwen-7B for chain-of-thought analysis | Free |
| **Orchestration** | LangGraph (state machine, multi-agent, HITL) | Free |
| **Document Intelligence** | PageIndex (tree indexing) + PyMuPDF for PDF parsing | Free (local mode) |
| **Tool Integration** | MCP servers for SEC EDGAR, FRED, Finnhub, yfinance | Free |
| **Guardrails** | NeMo Guardrails (input/output/dialog rails) | Free |
| **Risk Taxonomy** | Custom F3-inspired taxonomy via `mitreattack-python` patterns | Free |
| **Data: Filings** | SEC EDGAR (unlimited, no key needed) | Free |
| **Data: Macro** | FRED API (800K+ series, free key) | Free |
| **Data: Market** | Finnhub (60 req/min free) + yfinance | Free |
| **Data: Fundamentals** | Financial Modeling Prep (250 req/day free) | Free |
| **Vector Store** | ChromaDB (local, for hybrid search fallback) | Free |
| **UI** | Streamlit | Free |
| **Deployment** | Docker + Ollama on M4 MacBook Air (16GB) | Free |

### Four specialized agents orchestrated by LangGraph

**Agent 1: Document Intelligence Agent.** This agent owns SEC filing analysis. It uses PageIndex to build hierarchical tree indexes of 10-K, 10-Q, and 8-K filings, then navigates those trees using reasoning-based retrieval. When a user asks "What are Apple's key risk factors related to supply chain?", this agent navigates the tree to Item 1A (Risk Factors), extracts relevant paragraphs, follows cross-references to related notes, and returns structured findings with exact page citations. For tables (balance sheets, income statements), it leverages PageIndex's vision-native mode, sending PDF page images directly to the LLM for visual interpretation — preserving row-column relationships that OCR destroys. Every finding includes the exact tree navigation path for full auditability.

**Agent 2: Quantitative Analysis Agent.** This agent handles numerical financial analysis. It pulls real-time and historical data via MCP servers connected to Finnhub, yfinance, FRED, and Financial Modeling Prep. It computes financial ratios (P/E, debt-to-equity, current ratio, free cash flow yield), performs trend analysis across quarters, runs comparative analysis against sector benchmarks, and correlates company metrics with macroeconomic indicators from FRED. The RLM paradigm enables this agent to programmatically write and execute Python code in a REPL environment for financial calculations — computing DCF models, running regression analysis, or building simple factor models directly within the reasoning loop.

**Agent 3: Risk Classification Agent (F3 Taxonomy).** This is FinSight's most novel component. Inspired by MITRE's F3 framework for financial fraud, this agent maintains a structured taxonomy of financial risk patterns organized by tactic and technique:

- **Disclosure Risk Tactics**: Omission (missing required disclosures), Obfuscation (complex language hiding material changes), Inconsistency (contradictions between filing sections)
- **Financial Health Tactics**: Revenue manipulation (channel stuffing language, bill-and-hold patterns), Expense deferral (capitalization of operating expenses), Off-balance-sheet risks (special purpose entity references)
- **Governance Tactics**: Related-party transaction patterns, executive compensation red flags, auditor change signals
- **Market Risk Tactics**: Concentration risk (customer/supplier dependency), Regulatory exposure, Litigation trajectory

The agent scans filings for linguistic and structural patterns matching each technique, assigns confidence scores, and produces a "risk heat map" with ATT&CK-style technique IDs (e.g., DR-T001: Material omission in risk factors). This structured classification is something no existing open-source project provides.

**Agent 4: Synthesis & Report Agent.** The final agent aggregates findings from all three upstream agents, resolves contradictions using RLM-powered cross-document reasoning, and generates a structured investment analysis report. The report includes an executive summary, key financial metrics with trends, risk classification heat map, comparative analysis, and a bull/bear thesis with supporting evidence and citations. This agent uses the RLM library to process findings across multiple documents — comparing this quarter's 10-Q language against the previous year's 10-K, identifying material changes in risk factor language, and synthesizing multi-year trends.

### How guardrails make this production-ready

The guardrails architecture operates at five levels, making FinSight uniquely suitable for demonstrating production AI safety:

**Input rails** validate user queries, reject prompt injection attempts, and mask any PII (account numbers, SSNs) before they reach the LLM. **Output rails** enforce structured response formats, detect and flag potential hallucinations by cross-referencing claims against retrieved source material, and ensure no financial advice is stated without qualification. **Retrieval rails** filter PageIndex tree search results for relevance and flag when retrieved content may be outdated (e.g., using a prior year's filing for a current-year question). **Execution rails** constrain tool access — the system can read SEC filings but cannot execute trades or access non-public data. **Human-in-the-loop breakpoints** pause the workflow when the Risk Classification Agent assigns a high-confidence fraud signal, requiring human review before including it in the final report. LangGraph's `interrupt_before` mechanism handles this natively.

An **adaptive feedback loop** logs every guardrail decision, user correction, and agent self-correction, creating data for continuous improvement. An **emergency stop** circuit breaker terminates any agent loop exceeding 20 iterations or consuming excessive tokens.

### How RLM and PageIndex work together uniquely

This combination is FinSight's deepest technical innovation. PageIndex transforms a 200-page 10-K filing into a hierarchical JSON tree (~2-5KB), which fits easily in any LLM's context window. The RLM paradigm then enables reasoning across *collections* of these trees. When comparing Apple's risk factors across five years of 10-K filings, the RLM treats all five tree structures as external environment variables. The root LM writes Python code to grep across trees, identify nodes with matching titles (e.g., "Risk Factors" across years), extract the relevant page ranges from each filing, and then recursively call sub-LMs to analyze each year's content. This enables multi-document, multi-year analysis that would be impossible with a standard context window — even a 128K token window cannot hold five full 10-K filings simultaneously.

The implementation uses the official `rlm` Python library (`pip install rlm`, MIT license, 2.6K GitHub stars). The RLM-Qwen3-8B post-trained model outperforms base Qwen3-8B by **28.3%** on average, and can be run locally via Ollama on the M4 MacBook Air.

### What this demonstrates to hiring managers

FinSight is designed to check every box a 2026 AI hiring manager looks for:

- **MCP integration**: Custom MCP servers for SEC EDGAR, FRED, Finnhub — demonstrates protocol fluency
- **LangGraph orchestration**: Multi-agent state machine with conditional routing, parallel execution, and feedback loops
- **Agentic RAG**: PageIndex tree-based retrieval with corrective fallback to ChromaDB vector search
- **Guardrails**: Full NeMo Guardrails integration with input/output/retrieval/execution/HITL rails
- **Production deployment**: Docker containerized, Ollama-served local LLM, Streamlit UI, full observability logging
- **Domain expertise**: Deep financial domain knowledge (SEC filings, XBRL, financial ratios, risk analysis)
- **Novel research integration**: RLM paper implementation, MITRE F3-inspired taxonomy — shows the candidate reads and implements cutting-edge research
- **End-to-end ownership**: From data ingestion to deployed application with monitoring

---

## Part 3: Why FinSight is genuinely novel

### Systematic comparison against existing projects

| Feature | FinGPT | FinRobot | FinRL | OpenBB | **FinSight** |
|---|---|---|---|---|---|
| Layout-aware document analysis | ✗ | Partial | ✗ | ✗ | **✓ (PageIndex)** |
| Multi-agent orchestration | ✗ | Basic (AutoGen) | ✗ | ✗ | **✓ (LangGraph)** |
| Risk taxonomy classification | ✗ | ✗ | ✗ | ✗ | **✓ (F3-inspired)** |
| Guardrails + HITL | ✗ | ✗ | ✗ | ✗ | **✓ (NeMo)** |
| Runs fully local | Partial | ✗ (needs GPT-4) | ✓ | ✓ | **✓ (Qwen3 8B)** |
| Cross-document reasoning | ✗ | ✗ | ✗ | ✗ | **✓ (RLM)** |
| MCP tool integration | ✗ | ✗ | ✗ | ✓ | **✓** |
| Audit trail / explainability | ✗ | ✗ | ✗ | ✗ | **✓** |
| Free to run | ✓ | ✗ ($$ API) | ✓ | ✓ | **✓** |

FinSight occupies a unique position: it is the only system combining layout-aware document intelligence, structured risk classification, multi-agent orchestration with guardrails, and cross-document reasoning — all running locally on consumer hardware.

### Three specific novelty claims

**First**, no existing open-source project applies PageIndex's tree-based RAG to financial documents in a multi-agent setting. PageIndex has demonstrated 98.7% accuracy on FinanceBench with its Mafin 2.5 system, but this capability has not been integrated into a broader financial analysis pipeline with risk classification and quantitative analysis agents. FinSight is the first to use PageIndex as the document intelligence backbone of a multi-agent financial analyst.

**Second**, the MITRE F3-inspired financial risk taxonomy as an AI classification system is entirely new. MITRE only announced the F3 framework in May 2025, and no open-source implementation exists. FinSight would be the first project to operationalize this concept — using LLM-based analysis to classify language patterns in SEC filings against a structured taxonomy of financial fraud and risk techniques.

**Third**, the RLM + PageIndex combination for cross-document financial reasoning has never been implemented. RLM was published in December 2025, and no financial application exists yet. Using RLM to reason across collections of PageIndex tree structures — enabling multi-year, multi-company financial analysis without context window limitations — is a genuinely new architectural pattern.

### How FinSight complements FinSent-CoT

The two projects form a natural pipeline with zero overlap. **FinSent-CoT** operates at the sentence/paragraph level, performing chain-of-thought sentiment classification on financial text (positive/negative/neutral with reasoning). **FinSight** operates at the document and cross-document level, performing structural analysis of complex financial filings, extracting quantitative data, and classifying systemic risks. FinSent-CoT answers "What is the sentiment of this earnings call paragraph?" FinSight answers "What are the material risk changes across Apple's last three 10-K filings, and which match known fraud/risk patterns?" Together, they demonstrate both NLP depth (FinSent-CoT) and systems engineering breadth (FinSight).

### Implementation roadmap

A realistic build sequence for a motivated developer:

**Week 1-2:** Set up infrastructure — Ollama with Qwen 3 8B Q4_K_M, basic LangGraph skeleton, MCP servers for SEC EDGAR and FRED, Streamlit shell. Get a single agent retrieving and parsing a 10-K filing using PageIndex's local mode.

**Week 3-4:** Build the Document Intelligence Agent with PageIndex integration. Implement tree generation for SEC filings, reasoning-based retrieval, and basic cross-reference following. Add ChromaDB as a fallback vector store for hybrid search.

**Week 5-6:** Build the Quantitative Analysis Agent with MCP-connected data sources. Implement financial ratio computation, trend analysis, and FRED macro correlation. Integrate with LangGraph's shared state.

**Week 7-8:** Build the Risk Classification Agent with the F3-inspired taxonomy. Define the taxonomy structure, implement pattern matching against filing language, build the risk heat map visualization.

**Week 9-10:** Build the Synthesis Agent with RLM integration for cross-document reasoning. Implement the `rlm` library for multi-filing analysis. Build the final report generation pipeline.

**Week 11-12:** Add guardrails (NeMo Guardrails for input/output/retrieval rails, HITL breakpoints), observability logging, Docker containerization, comprehensive README with architecture diagrams, and demo video.

---

## Conclusion: positioning for maximum hiring impact

The 2026 AI job market has crystallized around a clear signal: **production-grade agentic AI systems with safety and observability are what matter**. MCP, LangGraph, agentic RAG, and guardrails have moved from bleeding-edge to baseline expectations in under a year. The candidates who stand out are those who build real systems that integrate these technologies in meaningful domains — not tutorial recreations, but novel architectures that solve genuine problems.

FinSight is designed to be that system. It addresses the single largest gap in the open-source finance AI landscape (explainable, auditable document intelligence with guardrails), integrates every technique that appears in 2026 job postings (MCP, LangGraph, agentic RAG, guardrails, HITL), and does so on consumer hardware with zero cost. The MITRE F3-inspired risk taxonomy and RLM integration provide genuine research novelty that goes beyond engineering — demonstrating the kind of original thinking that Anthropic, OpenAI, and top AI labs explicitly prioritize over credentials.

Combined with FinSent-CoT for sentiment analysis, FinSight creates a portfolio that covers the full stack of modern AI engineering: from fine-tuning and NLP (FinSent-CoT) to multi-agent systems, document intelligence, and production safety (FinSight). That combination — depth in one area, breadth across the system — is precisely what separates candidates who get hired from those who don't.