# ACID Benchmark: Comprehensive Project Context & Architecture Reference

## 1. Executive Summary & Core Philosophy

* **The Problem:** Modern LLMs boast massive context windows (ranging from 128k to millions of tokens), but scale does not equal comprehension. Most existing benchmarks evaluate models under pristine, zero-shot conditions or test simple retrieval ("needle-in-a-haystack"). They fail to measure how *pure logical reasoning and multi-step deduction* degrade when the context is saturated with real-world bloat (e.g., massive system prompts, chat history, dense RAG payloads, or logs).
* **The Solution:** **ACID** measures **context robustness** via a relative conservation ratio. Instead of rewarding raw intelligence in a vacuum, it isolates and scores how well a model *retains* its baseline reasoning capability as background noise scales exponentially.

---

## 2. Mathematical Framework & Metrics

### A. The Linear Conservation Ratio

For any given model, question, and noise tier $T$, the performance is scored relative to its own clean-context baseline:

$$\text{Conservation Ratio}_{T} = \frac{\text{Score}_{\text{noise}(T)}}{\text{Score}_{\text{baseline}}}$$

* *Why Linear Ratio:* Keeps scoring transparent, intuitive, and easy to interpret. Visual degradation curves are handled separately via logarithmic scaling on the axis.

### B. The ACID Compliance Thresholds (ACID80 / ACID90)

Rather than relying on a single complex composite score, ACID utilizes performance-cliff thresholds similar to technical standards:

* **ACID80:** The maximum noise token tier where a model's conservation ratio remains at or above $0.80$ (representing the boundary of practical operational usability).
* **ACID90:** The high-reliability standard, tracking where performance drops below $0.90$.

---

## 3. Experimental Design & Methodology

### A. Noise Corpus & Preprocessing Constraints

* **Content:** Disjoint, non-threatening humanities, history, or general Wikipedia text blocks.
* **Size Tiers:** Tested across discrete, exponential brackets: **0 Tokens (Baseline)** $\rightarrow$ **8K** $\rightarrow$ **32K** $\rightarrow$ **128K** $\rightarrow$ **512K** tokens.
* **The Trailing Question Rule:** All noise blocks must have any trailing question marks or explicit interrogation syntax strictly stripped from their endpoints. This prevents accidental prompt hijacking or task confusion.

### B. Prompt Architecture

* **The Anchor Principle:** The noise block is prepended as a massive prefix, and the actual test question/prompt is securely anchored at the **very end** of the input payload:
`[ Massive Static Noise Block (e.g., 128k tokens) ] + [ Target Question (Math/Code/Bio/Chem) ]`
* **Static Reuse:** Because the noise block for a given tier is identical across all trials and domains, the evaluation harness should reuse pre-compiled static files. This maximizes provider-side **prompt caching** and drastically reduces API expenditure.

---

## 4. Domain Sub-Benchmarks

To uncover structural vulnerabilities, ACID categorizes evaluation items into four distinct tracks:

1. **ACID-Math:** Multi-step calculation chains, physics problem constraints, and variable tracking.
2. **ACID-Chem:** Stoichiometry, reaction mechanisms, and molecular nomenclature stability.
3. **ACID-Bio:** Systemic interactions, hierarchical classifications, and pathway logic.
4. **ACID-Code:** Syntax retention, variable scope evaluation, and algorithmic generation.

---

## 5. Intended Implementation Architecture for the Agent

When building the repository, the automated agent should structure the code into these modular components:

* `generator.py`: Handles noise corpus fetching, cleaning (stripping trailing questions), and token-budgeted file compilation.
* `harness.py`: Asynchronous execution engine that manages API calls (via OpenAI/OpenRouter compatible endpoints), handles prompt caching strategies, and loops through baseline and tiered noise payloads.
* `metrics.py`: Computes the conservation ratios, slices data by domain metadata tags (`ACID-Math`, `ACID-Code`, etc.), and extracts ACID80/ACID90 thresholds.
* `dataset_schema.json`: Standardized JSON schema ensuring every test item includes a unique ID, domain tag, problem statement, and evaluation criteria/answer key.
