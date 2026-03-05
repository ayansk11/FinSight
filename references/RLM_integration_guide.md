# RLM works with Qwen3.5 — here's the full picture

**The RLM library is fully model-agnostic and will work with Qwen3.5 out of the box.** Swapping Qwen3 for Qwen3.5 requires changing a single model name string — no code modifications, no architecture-specific dependencies. Qwen3.5 launched in February–March 2026 with models from 0.8B to 397B parameters, and the **9B dense model at Q4_K_M quantization (6.6 GB) is the ideal choice for a MacBook Air M4 with 16GB RAM** via Ollama. However, no fine-tuned RLM-Qwen3.5 model exists yet, and Qwen3.5's strong native 256K context window may reduce the marginal benefit of RLM scaffolding compared to shorter-context models.

## Qwen3.5 is a full model family with a new architecture

Qwen3.5 rolled out in three waves: the **397B-A17B flagship** on February 16, the **medium series** (122B-A10B, 35B-A3B, 27B) on February 24, and the **small series** (0.8B, 2B, 4B, 9B) on March 2, 2026. All open-weight models carry an **Apache 2.0 license**.

The architecture represents a significant departure from Qwen3. Where Qwen3 used standard transformers, Qwen3.5 employs a **hybrid Gated Delta Networks + sparse MoE design** — most layers use linear attention (Gated DeltaNet) for efficiency, with every fourth layer using standard gated attention for expressiveness. All models are **natively multimodal** (text, image, video processed in the same latent space via early fusion), support **256K native context** (extendable to 1M+ via YaRN), and include built-in thinking mode with `<think>` tags. Code generation benchmarks are strong: **83.6% on LiveCodeBench v6** and **76.4% on SWE-Bench Verified** for the flagship model.

The small models punch well above their weight. Qwen3.5-9B surpasses GPT-OSS-120B on GPQA Diamond (81.7 vs 71.5) and beats GPT-5-Nano on vision tasks, making it a remarkably capable model for its size class.

## The RLM library treats models as interchangeable black boxes

The RLM library (2.7K GitHub stars, MIT license) uses a clean **client abstraction layer** where models are accessed exclusively through standard chat completion APIs. The `OpenAIClient` class connects to any OpenAI-compatible endpoint — including vLLM, Ollama, and cloud APIs — using just a `model_name` string and optional `base_url`. There are **no hardcoded model names, no model-specific tokenizers, no Qwen-specific code** anywhere in the codebase.

Switching from Qwen3 to Qwen3.5 is trivial:

```python
from rlm import RLM
# Point at a local Ollama endpoint running Qwen3.5
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "qwen3.5",
        "base_url": "http://localhost:11434/v1"
    },
)
```

The core RLM mechanism requires only that the model can: follow a system prompt, generate Python code, handle iterative multi-turn REPL interactions, and produce structured final answers. These are **standard capabilities of any modern instruction-tuned LLM**, and Qwen3.5 excels at all of them.

One practical nuance deserves attention. The RLM paper's Appendix A explicitly notes that **system prompts optimized for one model family may produce "undesirable behavior" with another**. The prompts originally written for GPT-5 needed adjustment for Qwen3-Coder. Qwen3.5's different architecture and thinking mode behavior (controlled via API parameters rather than `/think` soft switches) means you should expect to **tune the RLM system prompt** for optimal performance with Qwen3.5.

## Running locally: Qwen3.5-9B fits comfortably on 16GB

Ollama officially supports Qwen3.5 as of version **0.17.5** (released March 2, 2026). Earlier versions had critical bugs including architecture detection failures and repetition issues — **updating to v0.17.5+ is mandatory**.

For a MacBook Air M4 with 16GB RAM, these models are viable:

| Model | Quantization | Size | Verdict |
|---|---|---|---|
| **qwen3.5:9b** | Q4_K_M | **6.6 GB** | **Best choice** — leaves ~9 GB for OS and context |
| qwen3.5:9b | Q8_0 | 11 GB | Fits but tight; less context headroom |
| qwen3.5:4b | Q4_K_M | 3.4 GB | Good fallback for longer contexts |
| qwen3.5:27b | Q4_K_M | 17 GB | Does NOT fit — causes heavy swap thrashing |

The **9B Q4_K_M model is the clear winner**: it consumes ~6.6 GB at rest, leaving ample room for the operating system and context windows up to 32K tokens (~10–11 GB total). Expected inference speed is **15–25 tokens/second** via Ollama's llama.cpp backend. For roughly **2× faster generation**, consider MLX via LM Studio instead — it's purpose-built for Apple's unified memory architecture.

Unsloth provides extensive GGUF quantizations on HuggingFace for all Qwen3.5 sizes, including their Dynamic 2.0 formats (UD-Q4_K_XL, UD-Q3_K_XL). However, **Unsloth's custom Dynamic formats may not work in Ollama** and require llama.cpp directly. The standard Ollama library models work reliably.

## No RLM-Qwen3.5 fine-tuned model exists yet

The only official fine-tuned RLM model is **RLM-Qwen3-8B** (`mit-oasys/rlm-qwen3-8b-v0.1`), which improved base Qwen3-8B performance by **28.3% on average** across long-context benchmarks using just 1,000 training trajectories. Because Qwen3.5 uses an entirely different architecture (Gated DeltaNet hybrid vs. standard transformer), **these weights cannot transfer** to Qwen3.5 models.

Creating an RLM-Qwen3.5 fine-tuned model would require generating fresh training trajectories with Qwen3.5 and then running supervised fine-tuning. Unsloth supports Qwen3.5 fine-tuning via **bf16 LoRA** for the 0.8B through 27B sizes, though QLoRA (4-bit) is not recommended due to higher-than-normal quantization differences in Qwen3.5's architecture.

Community reproduction studies offer an important insight: a paper titled "Think, But Don't Overthink" tested RLM with DeepSeek v3.2 and Kimi K2, finding that **models with strong native long-context abilities actually performed worse with RLM scaffolding** (Kimi K2 dropped from 86.6% to 60.0%). Since Qwen3.5's native 262K context is substantially larger than Qwen3's, the incremental benefit of RLM may be smaller — though RLM still provides significant gains on **information-dense tasks requiring semantic reasoning across the full context**, which is precisely the FinSight use case.

## Practical implications for FinSight

For a financial document intelligence system using RLM on consumer hardware, the architecture should work as follows: Ollama serves Qwen3.5-9B locally as an OpenAI-compatible endpoint, and the RLM library connects to it for cross-document reasoning. The RLM scaffold excels here because financial analysis typically involves **comparing data points scattered across multiple documents** — exactly the pattern where RLM's REPL-based context manipulation outperforms naive context stuffing.

Three practical considerations shape this deployment:

- **Context window management matters more than model size.** At Q4_K_M, the 9B model leaves enough RAM for ~32K token context windows. For larger document sets, RLM's recursive decomposition is essential — it offloads context to the REPL environment rather than cramming everything into the attention window. This is the primary value of RLM for FinSight, even with Qwen3.5's large native context.

- **Prompt engineering is the main integration work.** The RLM system prompt needs tuning for Qwen3.5's response patterns. Financial domain-specific instructions (e.g., how to extract figures from SEC filings, cross-reference balance sheets, handle table data) should be embedded in the system prompt. The `famitzsy8/rvlm` community extension, which adds PDF and image support to RLM, may be a useful starting point for document processing.

- **vLLM is not yet an option for Qwen3.5 locally.** vLLM requires version 0.17.0+ for Qwen3.5 support, and running vLLM on a MacBook Air is impractical anyway. **Ollama is the correct local serving choice**, and the RLM library connects to it via the OpenAI-compatible client with `base_url` set to `http://localhost:11434/v1`.

## Conclusion

The RLM + Qwen3.5 combination is viable and straightforward for FinSight. The library's model-agnostic design means **zero code changes** are needed — only a model name swap and system prompt tuning. The **Qwen3.5-9B at Q4_K_M (6.6 GB)** is the optimal local model, offering benchmark performance that rivals models many times its size while fitting comfortably on 16GB hardware via Ollama v0.17.5+. The main gap is the absence of a fine-tuned RLM-Qwen3.5 model, but the vanilla scaffold approach still provides substantial benefits for multi-document financial reasoning. The most important early investment is crafting RLM system prompts tuned to Qwen3.5's response characteristics and optimized for financial document patterns.