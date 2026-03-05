# RLM Implementation in FinSight: Complete Technical Guide

## The Core Problem RLM Solves for FinSight

A single Apple 10-K filing is ~80,000 tokens. Five years of 10-K + 10-Q filings for one company easily exceeds **400,000+ tokens** — far beyond any local 8B model's context window (typically 8K-32K tokens). Even 128K-context models suffer severe "context rot" (degraded accuracy on information in the middle of long contexts).

**RLM's solution:** Instead of cramming documents into the context window, the LLM gets *only* the user's query. The documents are stored as Python variables in a REPL environment. The LLM then writes Python code to programmatically search, partition, and recursively analyze those documents — calling itself (or smaller sub-LMs) on specific chunks.

Think of it like this: instead of handing a human analyst a stack of 500 pages and saying "read all of this and answer my question," you give them the question and a computer with the documents loaded, and they write search queries, grep for specific sections, and read only what's relevant.

---

## Architecture: Where RLM Sits in FinSight

```
User Query: "How has Apple's supply chain risk language changed 
             across the last 3 years of 10-K filings?"

                         │
                         ▼
              ┌─────────────────────┐
              │  LangGraph Router   │
              │  (decides: this     │
              │  needs cross-doc    │
              │  reasoning)         │
              └─────┬───────────────┘
                    │
                    ▼
     ┌──────────────────────────────────┐
     │  Synthesis Agent (uses RLM)      │
     │                                  │
     │  1. Loads PageIndex trees for    │
     │     AAPL 10-K 2022, 2023, 2024  │
     │  2. Calls rlm.completion()       │
     │     with trees as context vars   │
     │  3. Root LM writes Python to:    │
     │     - grep for "supply chain"    │
     │     - extract matching sections  │
     │     - launch sub-LM calls on    │
     │       each year's sections       │
     │     - compare and synthesize     │
     │  4. Returns structured analysis  │
     └──────────────────────────────────┘
```

**RLM is NOT used for every query.** It's triggered specifically when:
- Cross-document reasoning is needed (multi-year, multi-company)
- A single document exceeds the local LLM's context window
- The query requires comparing specific sections across filings

For simple single-document queries (e.g., "What was Apple's revenue in 2024?"), the Document Intelligence Agent uses PageIndex tree navigation directly — no RLM needed.

---

## Implementation Option A: Official `rlm` Library (Recommended)

The official library by alexzhang13 supports local models via vLLM's OpenAI-compatible endpoint.

### Step 1: Local Model Setup with Ollama + vLLM-compatible endpoint

```bash
# Install Ollama (already on your M4 Mac)
ollama pull qwen3:8b-q4_K_M

# Ollama exposes an OpenAI-compatible API at localhost:11434
# The rlm library can interface with this via the OpenAI client
```

### Step 2: Install and Configure RLM

```bash
# Clone the official repo
git clone https://github.com/alexzhang13/rlm.git
cd rlm

# Install with uv (recommended by the project)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init && uv venv --python 3.12
uv pip install -e .

# OR with pip
pip install -e .
```

### Step 3: Basic RLM Call with Local Ollama Model

```python
from rlm import RLM

# Configure RLM to use local Ollama model via OpenAI-compatible API
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "base_url": "http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
        "model_name": "qwen3:8b-q4_K_M",
        "api_key": "ollama",  # Ollama doesn't need a real key
    },
    max_iterations=15,      # Max REPL iterations before forcing answer
    max_depth=1,            # Depth of recursive sub-calls
    verbose=True,           # See what the LLM is doing in the REPL
)

# Simple test
result = rlm.completion("What is 2 + 2?")
print(result.response)
```

### Step 4: Financial Document Analysis with RLM

Here's the actual FinSight pattern — loading PageIndex trees as context variables:

```python
import json
from rlm import RLM
from rlm.logger import RLMLogger

# ═══════════════════════════════════════════════════════════
# STEP 1: Generate PageIndex trees for SEC filings
# ═══════════════════════════════════════════════════════════

# PageIndex converts a 200-page PDF into a ~2-5KB JSON tree
# Each tree node has: title, page_range, children, summary
# Example tree structure for a 10-K:
pageindex_tree_2024 = {
    "title": "Apple Inc. 10-K Annual Report FY2024",
    "page_range": [1, 198],
    "children": [
        {
            "title": "Part I",
            "page_range": [5, 45],
            "children": [
                {"title": "Item 1 - Business", "page_range": [5, 20], "children": [...]},
                {"title": "Item 1A - Risk Factors", "page_range": [21, 38], "children": [
                    {"title": "Macroeconomic and Industry Risks", "page_range": [21, 26]},
                    {"title": "Supply Chain and Manufacturing", "page_range": [26, 30]},
                    {"title": "Legal and Regulatory", "page_range": [30, 35]},
                    {"title": "Financial Risks", "page_range": [35, 38]},
                ]},
                {"title": "Item 1B - Unresolved Staff Comments", "page_range": [38, 38]},
                {"title": "Item 2 - Properties", "page_range": [39, 40]},
            ]
        },
        {
            "title": "Part II",
            "page_range": [46, 150],
            "children": [
                {"title": "Item 7 - MD&A", "page_range": [46, 80], "children": [...]},
                {"title": "Item 8 - Financial Statements", "page_range": [80, 140], "children": [...]},
            ]
        }
    ]
}

# In practice, PageIndex generates these trees automatically:
# from pageindex import PageIndex
# pi = PageIndex(model="local")  # or API mode
# tree_2024 = pi.index("apple_10k_2024.pdf")
# tree_2023 = pi.index("apple_10k_2023.pdf")
# tree_2022 = pi.index("apple_10k_2022.pdf")


# ═══════════════════════════════════════════════════════════
# STEP 2: Load raw filing text (for the REPL environment)
# ═══════════════════════════════════════════════════════════

# The actual filing text is what the RLM will search through.
# PageIndex trees tell the RLM WHERE to look; the raw text
# is WHAT it reads.

def load_filing_text(filepath: str) -> str:
    """Load the full text of a SEC filing."""
    # In practice: use PyMuPDF or sec-api to get clean text
    with open(filepath, 'r') as f:
        return f.read()

filing_2024 = load_filing_text("data/apple_10k_2024.txt")  # ~80K tokens
filing_2023 = load_filing_text("data/apple_10k_2023.txt")
filing_2022 = load_filing_text("data/apple_10k_2022.txt")


# ═══════════════════════════════════════════════════════════
# STEP 3: Build the RLM context — the key innovation
# ═══════════════════════════════════════════════════════════

# Instead of stuffing all filings into the prompt, we create
# a structured context string that the REPL environment stores
# as a Python variable. The root LM gets ONLY the query + 
# instructions, and uses Python code to explore the context.

def build_rlm_context(filings: dict, trees: dict) -> str:
    """
    Build a structured context string for the RLM REPL.
    
    The RLM stores this entire string as `context` variable
    in the Python REPL. The root LM can then write code like:
        lines = context.split('\\n')
        # grep for specific sections
        # extract page ranges using tree metadata
        # launch sub-LM calls on extracted chunks
    """
    context_parts = []
    
    for year, filing_text in filings.items():
        tree = trees[year]
        context_parts.append(f"""
{'='*80}
DOCUMENT: {tree['title']}
YEAR: {year}
TOTAL PAGES: {tree['page_range'][1]}
{'='*80}

DOCUMENT INDEX (PageIndex Tree):
{json.dumps(tree, indent=2)}

{'='*80}
FULL TEXT:
{'='*80}
{filing_text}
""")
    
    return "\n\n".join(context_parts)

# Build the combined context
rlm_context = build_rlm_context(
    filings={"2024": filing_2024, "2023": filing_2023, "2022": filing_2022},
    trees={"2024": pageindex_tree_2024, "2023": pageindex_tree_2023, "2022": pageindex_tree_2022}
)

# This context could be 200K+ tokens — way beyond any context window
# But the RLM REPL stores it as a variable, not in the prompt!


# ═══════════════════════════════════════════════════════════
# STEP 4: Configure and run the RLM query
# ═══════════════════════════════════════════════════════════

logger = RLMLogger(log_dir="./logs/rlm_trajectories")

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "base_url": "http://localhost:11434/v1",
        "model_name": "qwen3:8b-q4_K_M",
        "api_key": "ollama",
    },
    max_iterations=20,
    max_depth=1,
    logger=logger,
    verbose=True,
    
    # Custom system prompt that teaches the RLM about financial docs
    custom_system_prompt="""You are a financial document analyst. You have access to 
    SEC filings stored in the `context` variable. The context contains:
    1. PageIndex trees (JSON) showing the document structure with page ranges
    2. Full text of each filing
    
    Strategy for analyzing financial documents:
    - First, parse the PageIndex tree JSON to understand document structure
    - Use the tree to identify relevant sections and their page ranges  
    - Use Python string operations (grep, split, slice) to extract those sections
    - Launch sub-LM calls (llm_query()) on extracted sections for detailed analysis
    - Compare findings across years systematically
    
    Always cite specific sections (e.g., "Item 1A, pages 21-38") in your analysis.
    Format your final answer with clear year-by-year comparisons."""
)

# The actual query — note how the context is passed separately
query = """Analyze how Apple's supply chain risk language has evolved 
across the 2022, 2023, and 2024 10-K filings. Specifically:
1. What new supply chain risks were added each year?
2. What risks were removed or downgraded?
3. How did the language intensity change (more cautious, less cautious)?
Provide specific quotes and page references."""

# This is the magic call — context goes to REPL, query goes to LM
result = rlm.completion(f"{query}\n\nContext:\n{rlm_context}")
print(result.response)

# Access the full trajectory for audit logging
if result.metadata:
    trajectory = result.metadata
    # Log to FinSight's audit trail
    save_audit_trail(query, trajectory, result.response)
```


# ═══════════════════════════════════════════════════════════
# WHAT HAPPENS INSIDE THE RLM (the REPL execution trace)
# ═══════════════════════════════════════════════════════════

# When you call rlm.completion(), here's what the root LM
# actually does inside the REPL environment. This is the LLM
# writing and executing Python code:

"""
--- Iteration 1 (Root LM writes code) ---

```python
import json

# Parse the context to find document boundaries
docs = context.split('=' * 80)
# Find PageIndex trees
trees = {}
for i, chunk in enumerate(docs):
    if 'DOCUMENT INDEX' in chunk:
        # Extract the JSON tree
        json_start = chunk.index('{')
        json_end = chunk.rindex('}') + 1
        tree_json = chunk[json_start:json_end]
        year = chunk.split('YEAR:')[1].split('\n')[0].strip()
        trees[year] = json.loads(tree_json)

# Find risk factor sections in each year's tree
for year, tree in trees.items():
    for part in tree['children']:
        for item in part.get('children', []):
            if 'Risk Factors' in item.get('title', ''):
                print(f"Year {year}: Risk Factors at pages {item['page_range']}")
                for sub in item.get('children', []):
                    if 'Supply' in sub.get('title', ''):
                        print(f"  -> Supply Chain section: pages {sub['page_range']}")
```

Output:
Year 2024: Risk Factors at pages [21, 38]
  -> Supply Chain section: pages [26, 30]
Year 2023: Risk Factors at pages [19, 35]
  -> Supply Chain section: pages [24, 28]
Year 2022: Risk Factors at pages [18, 33]
  -> Supply Chain section: pages [22, 26]


--- Iteration 2 (Root LM extracts supply chain sections) ---

```python
# Extract the supply chain text for each year
# Using page markers in the full text to locate sections

def extract_section(full_text, start_page, end_page):
    # PageIndex adds page markers like [PAGE 26] in the text
    start_marker = f"[PAGE {start_page}]"
    end_marker = f"[PAGE {end_page + 1}]"
    
    start_idx = full_text.find(start_marker)
    end_idx = full_text.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        # Fallback: use approximate character positions
        chars_per_page = len(full_text) // tree['page_range'][1]
        start_idx = chars_per_page * start_page
        end_idx = chars_per_page * end_page
    
    return full_text[start_idx:end_idx]

# Get supply chain sections
supply_2024 = extract_section(filing_texts['2024'], 26, 30)
supply_2023 = extract_section(filing_texts['2023'], 24, 28)
supply_2022 = extract_section(filing_texts['2022'], 22, 26)

print(f"2024 supply chain section: {len(supply_2024)} chars")
print(f"2023 supply chain section: {len(supply_2023)} chars")
print(f"2022 supply chain section: {len(supply_2022)} chars")
```

Output:
2024 supply chain section: 12847 chars
2023 supply chain section: 11203 chars
2022 supply chain section: 10456 chars


--- Iteration 3 (Root LM launches RECURSIVE sub-LM calls) ---

```python
# THIS IS THE KEY RLM FEATURE: launching sub-LM calls
# The root LM calls llm_query() which spawns a depth=1 LM
# that gets the extracted section in its context window

analysis_2024 = llm_query(
    f"Analyze this supply chain risk section from Apple's 2024 10-K. "
    f"List every specific risk mentioned, with exact quotes. "
    f"Rate the language intensity (1-5 scale).\n\n{supply_2024}"
)

analysis_2023 = llm_query(
    f"Analyze this supply chain risk section from Apple's 2023 10-K. "
    f"List every specific risk mentioned, with exact quotes. "
    f"Rate the language intensity (1-5 scale).\n\n{supply_2023}"
)

analysis_2022 = llm_query(
    f"Analyze this supply chain risk section from Apple's 2022 10-K. "
    f"List every specific risk mentioned, with exact quotes. "
    f"Rate the language intensity (1-5 scale).\n\n{supply_2022}"
)

print("2024:", analysis_2024)
print("2023:", analysis_2023)
print("2022:", analysis_2022)
```

# Each llm_query() call sends ~3-4K tokens (one section)
# to a sub-LM, which easily fits in the context window.
# The sub-LM returns a structured analysis of that section.


--- Iteration 4 (Root LM synthesizes and produces final answer) ---

```python
# Now the root LM has structured analyses of all three years
# It can compare them and produce the final synthesis

comparison = llm_query(
    f"Compare these three analyses of Apple's supply chain risk language "
    f"across 2022-2024. Identify: new risks added each year, risks removed, "
    f"and changes in language intensity.\n\n"
    f"2022 Analysis:\n{analysis_2022}\n\n"
    f"2023 Analysis:\n{analysis_2023}\n\n"  
    f"2024 Analysis:\n{analysis_2024}"
)

# Set the final answer
FINAL_ANSWER = comparison
```

# The REPL detects FINAL_ANSWER and returns it as result.response
"""


# ═══════════════════════════════════════════════════════════
# IMPLEMENTATION OPTION B: Minimal/Custom RLM (Lightweight)
# ═══════════════════════════════════════════════════════════

# If the full rlm library has compatibility issues with Ollama,
# you can implement the core RLM pattern yourself using the
# rlm-minimal reference. This is ~200 lines of Python.

class FinSightRLM:
    """
    Minimal RLM implementation for FinSight.
    Based on alexzhang13/rlm-minimal.
    
    Core idea:
    1. Store documents as Python variables (not in LLM prompt)
    2. Give LLM a Python REPL to explore the documents
    3. LLM writes code to grep, partition, extract sections
    4. LLM can call sub_llm_query() for recursive analysis
    5. LLM sets FINAL_ANSWER when done
    """
    
    def __init__(self, llm_client, max_iterations=20):
        self.llm = llm_client  # Any LLM client (Ollama, etc.)
        self.max_iterations = max_iterations
        self.trajectory = []  # For audit trail
    
    def completion(self, query: str, context: str) -> dict:
        """
        Main RLM call.
        
        Args:
            query: The user's question
            context: The full document context (can be arbitrarily large)
        
        Returns:
            dict with 'response', 'trajectory', 'iterations'
        """
        # Initialize REPL environment with context as a variable
        repl_globals = {
            "context": context,
            "FINAL_ANSWER": None,
            "llm_query": self._make_sub_llm_callable(),
            "json": __import__("json"),
            "re": __import__("re"),
        }
        
        system_prompt = self._build_system_prompt(query)
        conversation = [{"role": "system", "content": system_prompt}]
        
        for iteration in range(self.max_iterations):
            # Get LLM's next code block
            response = self.llm.chat(conversation)
            conversation.append({"role": "assistant", "content": response})
            
            # Extract Python code from response
            code = self._extract_code(response)
            
            if code is None:
                # LLM produced final answer in text form
                return {
                    "response": response,
                    "trajectory": self.trajectory,
                    "iterations": iteration + 1
                }
            
            # Execute code in REPL
            output, error = self._execute_code(code, repl_globals)
            
            # Log to trajectory for audit trail
            self.trajectory.append({
                "iteration": iteration,
                "code": code,
                "output": output[:2000],  # Truncate for logging
                "error": error,
            })
            
            # Check if FINAL_ANSWER was set
            if repl_globals["FINAL_ANSWER"] is not None:
                return {
                    "response": repl_globals["FINAL_ANSWER"],
                    "trajectory": self.trajectory,
                    "iterations": iteration + 1
                }
            
            # Feed execution result back to LLM
            result_msg = f"Code output:\n{output}" if not error else f"Error:\n{error}"
            conversation.append({"role": "user", "content": result_msg})
        
        return {
            "response": "Max iterations reached without final answer.",
            "trajectory": self.trajectory,
            "iterations": self.max_iterations
        }
    
    def _make_sub_llm_callable(self):
        """Create a function the REPL can call for sub-LM queries."""
        def llm_query(prompt: str) -> str:
            """Call a sub-LM on a smaller chunk of text."""
            response = self.llm.chat([
                {"role": "system", "content": "You are a financial document analyst. Analyze the provided text precisely and concisely."},
                {"role": "user", "content": prompt}
            ])
            return response
        return llm_query
    
    def _build_system_prompt(self, query: str) -> str:
        return f"""You are a recursive financial document analyst.

You have access to a Python REPL environment with:
- `context`: A variable containing the full text of financial documents (potentially 200K+ tokens)
- `llm_query(prompt)`: A function to launch sub-LM calls on smaller text chunks  
- `json` and `re` modules
- `FINAL_ANSWER`: Set this variable to your final answer when done

Your task: {query}

STRATEGY:
1. First, explore the `context` variable structure using Python (len, grep, split)
2. Use PageIndex tree metadata (JSON) in the context to find relevant sections
3. Extract specific sections using string slicing or regex
4. Use `llm_query()` to analyze individual sections (keep chunks under 4000 chars)
5. Synthesize findings across sections/documents
6. Set `FINAL_ANSWER = "your complete answer"` when done

Write Python code in ```python blocks. You'll see the output after each execution.
IMPORTANT: Always set FINAL_ANSWER when you have your complete analysis."""
    
    def _extract_code(self, response: str) -> str | None:
        """Extract Python code block from LLM response."""
        if "```python" in response:
            start = response.index("```python") + 9
            end = response.index("```", start)
            return response[start:end].strip()
        return None
    
    def _execute_code(self, code: str, globals_dict: dict) -> tuple[str, str]:
        """Execute Python code in the REPL and capture output."""
        import io, sys
        
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        error = ""
        
        try:
            exec(code, globals_dict)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
        
        sys.stdout = old_stdout
        return captured.getvalue(), error


# ═══════════════════════════════════════════════════════════
# INTEGRATION WITH LANGGRAPH
# ═══════════════════════════════════════════════════════════

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class FinSightState(TypedDict):
    """Shared state across all agents."""
    query: str
    filing_paths: list[str]
    pageindex_trees: dict          # {year: tree_json}
    doc_intel_findings: dict       # From Document Intelligence Agent
    quant_findings: dict           # From Quantitative Agent
    risk_classifications: dict     # From Risk Agent (F3 taxonomy)
    synthesis: str                 # From Synthesis Agent (RLM-powered)
    needs_cross_doc: bool          # Router flag
    audit_trail: list[dict]        # Full execution log

def route_query(state: FinSightState) -> str:
    """
    Router node: decides if query needs RLM cross-document reasoning.
    
    Triggers RLM when:
    - Query references multiple years/filings
    - Query asks for trends, changes, comparisons
    - Total document size exceeds single-context capacity
    """
    query = state["query"].lower()
    
    cross_doc_signals = [
        "compare", "changed", "evolved", "trend", "across",
        "year over year", "yoy", "previous", "historical",
        "last 3 years", "last 5 years", "multi-year",
        "2022 and 2023", "2023 and 2024",  # Explicit year comparisons
    ]
    
    if any(signal in query for signal in cross_doc_signals):
        return "synthesis_with_rlm"
    
    if len(state.get("filing_paths", [])) > 1:
        return "synthesis_with_rlm"
    
    return "synthesis_standard"

def synthesis_agent_rlm(state: FinSightState) -> FinSightState:
    """
    Synthesis Agent using RLM for cross-document reasoning.
    Called when multi-document analysis is needed.
    """
    from ollama import Client
    
    # Initialize the RLM (using custom implementation for Ollama)
    ollama_client = OllamaLLMClient(model="qwen3:8b-q4_K_M")
    rlm = FinSightRLM(llm_client=ollama_client, max_iterations=15)
    
    # Build context from PageIndex trees + raw text
    context = build_rlm_context(
        filings=load_all_filings(state["filing_paths"]),
        trees=state["pageindex_trees"]
    )
    
    # Enrich query with findings from other agents
    enriched_query = f"""
    Original question: {state['query']}
    
    Document Intelligence findings: {json.dumps(state.get('doc_intel_findings', {}))}
    Quantitative findings: {json.dumps(state.get('quant_findings', {}))}
    Risk classifications: {json.dumps(state.get('risk_classifications', {}))}
    
    Synthesize all findings and perform cross-document analysis.
    Resolve any contradictions between agents.
    Provide specific citations (filing year, item, page number).
    """
    
    # Run RLM
    result = rlm.completion(query=enriched_query, context=context)
    
    # Update state
    state["synthesis"] = result["response"]
    state["audit_trail"].append({
        "agent": "synthesis_rlm",
        "iterations": result["iterations"],
        "trajectory": result["trajectory"],
    })
    
    return state

def synthesis_agent_standard(state: FinSightState) -> FinSightState:
    """
    Standard synthesis for single-document queries.
    No RLM needed — direct LLM call with agent findings.
    """
    # Simple LLM call combining findings from other agents
    prompt = f"""
    Query: {state['query']}
    Document findings: {state.get('doc_intel_findings', {})}
    Quantitative data: {state.get('quant_findings', {})}
    Risk assessment: {state.get('risk_classifications', {})}
    
    Synthesize into a clear, cited analysis.
    """
    response = ollama_chat(prompt, model="qwen3:8b-q4_K_M")
    state["synthesis"] = response
    return state

# Build the LangGraph
workflow = StateGraph(FinSightState)

# Add nodes
workflow.add_node("doc_intel", document_intelligence_agent)
workflow.add_node("quant", quantitative_agent)
workflow.add_node("risk", risk_classification_agent)
workflow.add_node("router", route_query)
workflow.add_node("synthesis_rlm", synthesis_agent_rlm)
workflow.add_node("synthesis_standard", synthesis_agent_standard)

# Define edges
workflow.set_entry_point("doc_intel")
workflow.add_edge("doc_intel", "quant")
workflow.add_edge("quant", "risk")
workflow.add_edge("risk", "router")

# Conditional routing after risk agent
workflow.add_conditional_edges(
    "router",
    route_query,
    {
        "synthesis_with_rlm": "synthesis_rlm",
        "synthesis_standard": "synthesis_standard",
    }
)
workflow.add_edge("synthesis_rlm", END)
workflow.add_edge("synthesis_standard", END)

app = workflow.compile()


# ═══════════════════════════════════════════════════════════
# OLLAMA LLM CLIENT WRAPPER (for custom RLM)
# ═══════════════════════════════════════════════════════════

class OllamaLLMClient:
    """Wrapper around Ollama for use with FinSightRLM."""
    
    def __init__(self, model: str = "qwen3:8b-q4_K_M", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def chat(self, messages: list[dict]) -> str:
        """Send chat messages to Ollama and return response text."""
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_ctx": 8192,       # Context window for the LLM
                    "temperature": 0.1,     # Low temp for precise analysis
                    "num_predict": 2048,    # Max output tokens per call
                }
            }
        )
        return response.json()["message"]["content"]


# ═══════════════════════════════════════════════════════════
# PERFORMANCE CONSIDERATIONS ON M4 MACBOOK AIR
# ═══════════════════════════════════════════════════════════

"""
RLM Performance on MacBook Air M4 (16GB RAM):

Model: Qwen3 8B Q4_K_M (~5GB VRAM)
- Tokens/sec: ~20-30 tok/s on M4
- Each RLM iteration: 1-3 LLM calls (root + sub-calls)
- Average tokens per call: ~500-2000
- Time per iteration: 5-30 seconds

Typical cross-document query (3 years of 10-K filings):
- Iterations: 4-8
- Sub-LM calls: 3-6  
- Total LLM calls: ~10-15
- Total time: 2-5 minutes
- This is acceptable for a "deep analysis" feature

Single-document query (bypasses RLM):
- Direct PageIndex tree navigation + single LLM call
- Total time: 10-30 seconds

OPTIMIZATION STRATEGIES:
1. Cache PageIndex trees (generate once, reuse across queries)
2. Pre-extract common sections (Risk Factors, MD&A, Financial Statements)
3. Use smaller model (Qwen3 4B) for sub-LM calls — faster, still accurate for extraction
4. Limit max_iterations to 15 for cost/time control
5. Use async sub-calls when official rlm library adds support

MEMORY FOOTPRINT:
- Ollama with Qwen3 8B Q4: ~5GB RAM
- PageIndex trees: ~10-50KB per filing (negligible)
- Raw filing text in REPL: ~500KB per filing (in Python process memory)
- 3 filings loaded: ~1.5MB in REPL + 5GB for model = ~6.5GB total
- Well within 16GB M4 MacBook Air capacity
"""


# ═══════════════════════════════════════════════════════════
# AUDIT TRAIL INTEGRATION
# ═══════════════════════════════════════════════════════════

def save_audit_trail(query: str, rlm_result: dict, timestamp: str = None):
    """
    Save complete RLM execution trace for compliance/explainability.
    
    This is what makes FinSight unique — every analysis has a 
    fully traceable reasoning chain showing:
    1. What code the LLM wrote
    2. What sections it examined
    3. What sub-queries it launched
    4. How it synthesized the final answer
    """
    import datetime
    
    audit_entry = {
        "timestamp": timestamp or datetime.datetime.now().isoformat(),
        "query": query,
        "model": "qwen3:8b-q4_K_M",
        "iterations": rlm_result["iterations"],
        "trajectory": rlm_result["trajectory"],
        "final_response": rlm_result["response"],
        # Each trajectory step shows:
        # - The Python code the LLM wrote
        # - The output/error from execution
        # - Whether it launched sub-LM calls
        # This creates a complete "show your work" audit trail
    }
    
    # Save to JSONL for dashboard visualization
    with open("data/audit_trail.jsonl", "a") as f:
        f.write(json.dumps(audit_entry) + "\n")
    
    return audit_entry


# ═══════════════════════════════════════════════════════════
# EXAMPLE: FULL END-TO-END QUERY
# ═══════════════════════════════════════════════════════════

"""
User Query: "Compare Apple's debt position across 2022-2024, 
             including any changes in risk language about debt."

LangGraph Flow:
1. Doc Intel Agent → Uses PageIndex to extract debt-related sections
   from all 3 filings (Item 7 MD&A, Item 8 Notes to Financial Statements)
   
2. Quant Agent → Pulls actual debt figures via MCP:
   - Total debt, debt-to-equity ratio, interest coverage
   - From Financial Modeling Prep + yfinance
   
3. Risk Agent → Classifies debt-related risk language:
   - F3-T042: "Debt covenant compliance language changes"
   - F3-T043: "Interest rate risk disclosure intensity"
   
4. Router → Detects "across 2022-2024" → routes to RLM synthesis

5. Synthesis Agent (RLM) →
   - Receives all findings from agents 1-3
   - Loads all 3 filings as REPL context variables
   - Root LM writes code to:
     a. Extract debt-specific notes from each year's filing
     b. Launch sub-LM on each year's debt notes
     c. Compare quantitative data with language changes
     d. Synthesize: "Debt increased 12% YoY, but risk language 
        softened — the 2024 filing removed the phrase 'material 
        impact on liquidity' present in 2022-2023"
   - Sets FINAL_ANSWER with full cited analysis
   
6. Guardrails → Output rail validates:
   - Claims cross-referenced against retrieved source text
   - No financial advice stated without qualification
   - All citations verified against actual page numbers
   
Output: Structured report with audit trail
"""
