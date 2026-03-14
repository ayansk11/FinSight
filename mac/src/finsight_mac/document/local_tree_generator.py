"""Local PageIndex tree generator with Groq cloud fallback.

Generates hierarchical document structure trees from SEC filing PDFs
using heuristic heading detection + LLM refinement. Tries local Ollama
first; falls back to Groq cloud API when Ollama times out.

Flow:
  PDF → extract page texts → regex heading detection →
  LLM hierarchy refinement (Ollama → Groq fallback) →
  assign node IDs → optional summaries → validated PageIndexTree
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from finsight_core.models.document import PageIndexTree
from finsight_core.pageindex.parser import load_tree_from_dict
from finsight_core.prompts.tree_generation import (
    TREE_GEN_SYSTEM_PROMPT,
    build_hierarchy_prompt,
    build_summary_prompt,
)

logger = logging.getLogger(__name__)

# --- SEC filing heading patterns ---

# Level 1: PART headings
_RE_PART = re.compile(r"^\s*PART\s+(I{1,3}V?|IV|[1-4])\b", re.IGNORECASE)

# Level 2: Item headings
_RE_ITEM = re.compile(r"^\s*Item\s+\d+[A-Z]?\.?\s", re.IGNORECASE)

# Level 2: Other top-level sections
_RE_SIGNATURES = re.compile(r"^\s*SIGNATURES\s*$", re.IGNORECASE)
_RE_EXHIBITS = re.compile(r"^\s*EXHIBIT\s+(INDEX|LIST)", re.IGNORECASE)

# Skip list: lines that look like headings but aren't sections
_SKIP_PATTERNS = [
    re.compile(r"TABLE\s+OF\s+CONTENTS", re.IGNORECASE),
    re.compile(r"^\s*PAGE\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),  # Page numbers
    re.compile(r"UNITED\s+STATES", re.IGNORECASE),
    re.compile(r"SECURITIES\s+AND\s+EXCHANGE", re.IGNORECASE),
    re.compile(r"WASHINGTON", re.IGNORECASE),
    re.compile(r"FORM\s+10-[KQ]", re.IGNORECASE),
    re.compile(r"ANNUAL\s+REPORT", re.IGNORECASE),
    re.compile(r"QUARTERLY\s+REPORT", re.IGNORECASE),
]


class LocalTreeGenerator:
    """Generate PageIndex trees using PDF heuristics + LLM (Ollama → Groq).

    Tries local Ollama first. If it times out (common for large 100+ page
    filings on consumer hardware), automatically falls back to Groq cloud.

    Example:
        >>> gen = LocalTreeGenerator()
        >>> tree = await gen.generate("path/to/10k.pdf", "AAPL_10K_2024")
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._fallback_llm = None

    def _get_llm(self):
        """Lazy-init OllamaClient."""
        if self._llm is None:
            from finsight_mac.llm.ollama_client import OllamaClient

            self._llm = OllamaClient()
        return self._llm

    def _get_fallback_llm(self):
        """Lazy-init Groq fallback client (returns None if not configured)."""
        if self._fallback_llm is None:
            from finsight_mac.llm.groq_client import GroqClient

            if GroqClient.is_available():
                try:
                    self._fallback_llm = GroqClient()
                    logger.info("[LocalTreeGen] Groq fallback available")
                except Exception as e:
                    logger.debug(f"[LocalTreeGen] Groq init failed: {e}")
        return self._fallback_llm

    async def generate(
        self,
        pdf_path: str | Path,
        doc_name: str | None = None,
        add_summaries: bool = True,
        page_texts: dict[int, str] | None = None,
    ) -> PageIndexTree:
        """Generate a PageIndex tree from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            doc_name: Optional document name (defaults to filename stem).
            add_summaries: Whether to generate per-node summaries via Ollama.
            page_texts: Pre-extracted page texts (skips PDF extraction).

        Returns:
            Validated PageIndexTree ready for use.
        """
        pdf_path = Path(pdf_path)
        doc_name = doc_name or pdf_path.stem

        # Step 1: Extract page texts
        if page_texts is None:
            page_texts = self._extract_page_texts(pdf_path)

        max_page = max(page_texts.keys()) if page_texts else 0
        logger.info(f"[LocalTreeGen] Extracted text from {len(page_texts)} pages")

        # Step 2: Heuristic heading detection
        candidates = self._extract_heading_candidates(page_texts)
        logger.info(f"[LocalTreeGen] Found {len(candidates)} heading candidates")

        if not candidates:
            # Fallback: create a single root node spanning all pages
            tree_dict = {
                "doc_name": doc_name,
                "doc_description": f"Document: {doc_name}",
                "structure": [
                    {
                        "title": doc_name,
                        "node_id": "0000",
                        "start_index": 1,
                        "end_index": max_page or 1,
                        "summary": None,
                        "nodes": [],
                    }
                ],
            }
            return load_tree_from_dict(tree_dict)

        # Step 3: Build hierarchy (Ollama → Groq → level-based fallback)
        tree_dict = None

        # Try Ollama first
        try:
            tree_dict = await self._build_hierarchy_with_llm(
                self._get_llm(),
                "Ollama",
                candidates,
                doc_name,
                page_texts,
                max_page,
            )
        except Exception as e:
            logger.warning(f"[LocalTreeGen] Ollama hierarchy failed: {e}")

        # Try Groq fallback if Ollama failed
        if tree_dict is None:
            fallback = self._get_fallback_llm()
            if fallback is not None:
                try:
                    logger.info("[LocalTreeGen] Falling back to Groq cloud...")
                    tree_dict = await self._build_hierarchy_with_llm(
                        fallback,
                        "Groq",
                        candidates,
                        doc_name,
                        page_texts,
                        max_page,
                    )
                except Exception as e:
                    logger.warning(f"[LocalTreeGen] Groq hierarchy failed: {e}")

        # Final fallback: level-based heuristic (no LLM)
        if tree_dict is None:
            logger.warning("[LocalTreeGen] All LLMs failed. Using level-based fallback.")
            tree_dict = self._candidates_to_default_tree(candidates, doc_name, max_page)

        # Step 4: Assign node IDs and compute end indices
        structure = tree_dict.get("structure", [])
        self._assign_node_ids(structure)
        self._compute_end_indices(structure, max_page)

        # Step 5: Optional summaries
        if add_summaries:
            try:
                await self._generate_summaries(structure, page_texts)
            except Exception as e:
                logger.warning(f"[LocalTreeGen] Summary generation failed: {e}")

        # Step 6: Embed page texts in leaf nodes
        self._embed_leaf_texts(structure, page_texts)

        # Step 7: Validate with Pydantic
        final = {
            "doc_name": doc_name,
            "doc_description": tree_dict.get("doc_description", ""),
            "structure": structure,
        }
        tree = load_tree_from_dict(final)
        logger.info(f"[LocalTreeGen] Generated tree: {tree.total_nodes} nodes")
        return tree

    # -----------------------------------------------------------------
    # Phase 1: Heuristic heading extraction
    # -----------------------------------------------------------------

    def _extract_heading_candidates(self, page_texts: dict[int, str]) -> list[dict]:
        """Extract heading candidates from page text using regex.

        Returns:
            List of {"title": str, "page": int, "level": int} sorted
            by page number. Level 1 = PART, 2 = Item, 3 = sub-heading.
        """
        candidates: list[dict] = []
        seen_titles: set[str] = set()

        for page_num in sorted(page_texts.keys()):
            text = page_texts[page_num]
            lines = text.split("\n")

            for line in lines:
                stripped = line.strip()
                if not stripped or len(stripped) < 3:
                    continue

                # Skip known false positives
                if any(p.search(stripped) for p in _SKIP_PATTERNS):
                    continue

                candidate = self._classify_line(stripped, page_num)
                if candidate is None:
                    continue

                # Deduplicate (same title seen earlier)
                title_key = candidate["title"].lower().strip()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                candidates.append(candidate)

        return sorted(candidates, key=lambda c: c["page"])

    @staticmethod
    def _classify_line(line: str, page_num: int) -> dict | None:
        """Classify a single line as a heading candidate or None."""
        # Level 1: PART headings
        if _RE_PART.match(line):
            return {"title": line.strip(), "page": page_num, "level": 1}

        # Level 2: Item headings
        if _RE_ITEM.match(line):
            return {"title": line.strip(), "page": page_num, "level": 2}

        # Level 2: SIGNATURES
        if _RE_SIGNATURES.match(line):
            return {"title": "SIGNATURES", "page": page_num, "level": 2}

        # Level 2: EXHIBIT INDEX
        if _RE_EXHIBITS.match(line):
            return {"title": line.strip(), "page": page_num, "level": 2}

        # Level 3: ALL CAPS lines (potential sub-headings)
        if (
            line.isupper()
            and 5 <= len(line) <= 80
            and not line.startswith("(")
            and " " in line  # At least 2 words
        ):
            return {"title": line.strip(), "page": page_num, "level": 3}

        return None

    # -----------------------------------------------------------------
    # Phase 2: Hierarchy construction
    # -----------------------------------------------------------------

    async def _build_hierarchy_with_llm(
        self,
        llm,
        llm_name: str,
        candidates: list[dict],
        doc_name: str,
        page_texts: dict[int, str],
        max_page: int,
    ) -> dict:
        """Use an LLM to organize candidates into a nested tree.

        Args:
            llm: LLM client (OllamaClient or GroqClient).
            llm_name: Display name for logging (e.g., "Ollama", "Groq").
            candidates: Heading candidates from heuristic extraction.
            doc_name: Document name.
            page_texts: Page number → text mapping.
            max_page: Total page count.

        Returns:
            Dict with "doc_description" and "structure" keys.
        """
        # Build sample texts (first 150 chars per heading's page)
        sample_texts: dict[int, str] = {}
        for c in candidates[:30]:  # Limit to avoid context overflow
            pg = c["page"]
            if pg in page_texts and pg not in sample_texts:
                sample_texts[pg] = page_texts[pg][:150]

        prompt = build_hierarchy_prompt(candidates, doc_name, sample_texts)
        logger.info(f"[LocalTreeGen] Building hierarchy with {llm_name}...")
        result = await llm.achat_json(prompt, system_message=TREE_GEN_SYSTEM_PROMPT)

        # Convert LLM output format to our internal format
        doc_desc = result.get("doc_description", "")
        raw_structure = result.get("structure", [])

        structure = self._convert_llm_nodes(raw_structure)
        logger.info(f"[LocalTreeGen] {llm_name} produced {len(structure)} top-level nodes")

        return {
            "doc_description": doc_desc,
            "structure": structure,
        }

    def _convert_llm_nodes(self, llm_nodes: list) -> list[dict]:
        """Convert LLM JSON output to our internal node format."""
        result = []
        for node in llm_nodes:
            children = node.get("children", [])
            page = node.get("page", 1)
            converted = {
                "title": node.get("title", ""),
                "node_id": "",  # Assigned later
                "start_index": page,
                "end_index": page,  # Computed later
                "summary": None,
                "text": None,
                "nodes": self._convert_llm_nodes(children),
            }
            result.append(converted)
        return result

    def _candidates_to_default_tree(
        self,
        candidates: list[dict],
        doc_name: str,
        max_page: int,
    ) -> dict:
        """Build tree from level-based nesting without LLM.

        Level 1 (PART) nodes become top-level children of root.
        Level 2 (Item) nodes become children of the preceding PART.
        Level 3 nodes become children of the preceding Item.
        """
        root_children: list[dict] = []
        current_l1: dict | None = None
        current_l2: dict | None = None

        for c in candidates:
            node = {
                "title": c["title"],
                "node_id": "",
                "start_index": c["page"],
                "end_index": c["page"],
                "summary": None,
                "text": None,
                "nodes": [],
            }

            if c["level"] == 1:
                current_l1 = node
                current_l2 = None
                root_children.append(node)
            elif c["level"] == 2:
                current_l2 = node
                if current_l1 is not None:
                    current_l1["nodes"].append(node)
                else:
                    root_children.append(node)
            else:  # level 3
                if current_l2 is not None:
                    current_l2["nodes"].append(node)
                elif current_l1 is not None:
                    current_l1["nodes"].append(node)
                else:
                    root_children.append(node)

        # Wrap in a document root if we have content
        root = {
            "title": doc_name,
            "node_id": "",
            "start_index": 1,
            "end_index": max_page or 1,
            "summary": None,
            "text": None,
            "nodes": root_children,
        }

        return {
            "doc_description": f"SEC filing: {doc_name}",
            "structure": [root],
        }

    # -----------------------------------------------------------------
    # Phase 3: Node IDs, end indices, summaries
    # -----------------------------------------------------------------

    def _assign_node_ids(self, nodes: list[dict], counter: list[int] | None = None) -> None:
        """Assign sequential 4-digit DFS-order node IDs."""
        if counter is None:
            counter = [0]

        for node in nodes:
            node["node_id"] = f"{counter[0]:04d}"
            counter[0] += 1
            self._assign_node_ids(node.get("nodes", []), counter)

    def _compute_end_indices(self, nodes: list[dict], max_page: int) -> None:
        """Compute end_index for each node.

        end_index = next sibling's start_index - 1, or max_page for last.
        Parent's end_index spans all children.
        """
        for i, node in enumerate(nodes):
            # Determine end from next sibling
            if i + 1 < len(nodes):
                next_start = nodes[i + 1].get("start_index", max_page)
                end_idx = max(next_start - 1, node["start_index"])
            else:
                end_idx = max_page

            # Recurse into children first
            children = node.get("nodes", [])
            if children:
                self._compute_end_indices(children, end_idx)
                # Parent's end is at least the max of children's ends
                child_max = max(c.get("end_index", 0) for c in children)
                end_idx = max(end_idx, child_max)

            node["end_index"] = end_idx

    async def _generate_summaries(self, nodes: list[dict], page_texts: dict[int, str]) -> None:
        """Generate one-sentence summaries for all nodes.

        Tries Ollama first; falls back to Groq on failure.
        """
        llm = self._get_llm()
        # Pre-check: try one summary with Ollama; switch to Groq on timeout
        try:
            await llm.achat("Say 'ok'.", temperature=0)
        except Exception:
            fallback = self._get_fallback_llm()
            if fallback is not None:
                logger.info("[LocalTreeGen] Summaries: switching to Groq fallback")
                llm = fallback

        for node in nodes:
            start = node.get("start_index", 1)
            end = node.get("end_index", start)
            title = node.get("title", "")

            # Build text snippet (~500 chars from section pages)
            snippet_parts = []
            chars = 0
            for pg in range(start, min(end + 1, start + 3)):
                if pg in page_texts:
                    snippet_parts.append(page_texts[pg])
                    chars += len(page_texts[pg])
                    if chars > 500:
                        break

            snippet = "\n".join(snippet_parts)[:500]

            if snippet.strip():
                prompt = build_summary_prompt(title, snippet)
                try:
                    summary = await llm.achat(prompt, temperature=0.1)
                    node["summary"] = summary.strip()[:200]
                except Exception:
                    pass

            # Recurse into children
            children = node.get("nodes", [])
            if children:
                await self._generate_summaries(children, page_texts)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _embed_leaf_texts(self, nodes: list[dict], page_texts: dict[int, str]) -> None:
        """Embed concatenated page text into leaf nodes."""
        for node in nodes:
            children = node.get("nodes", [])
            if children:
                self._embed_leaf_texts(children, page_texts)
            elif not node.get("text"):
                start = node.get("start_index", 0)
                end = node.get("end_index", start)
                parts = []
                for pg in range(start, end + 1):
                    if pg in page_texts:
                        parts.append(page_texts[pg])
                if parts:
                    node["text"] = "\n\n".join(parts)

    @staticmethod
    def _extract_page_texts(pdf_path: Path) -> dict[int, str]:
        """Extract text from PDF pages using PyMuPDF."""
        from finsight_core.pageindex.text_extractor import (
            extract_pages_from_pdf,
        )

        return extract_pages_from_pdf(str(pdf_path))
