# Where Does the Image Matter? Translating and Evaluating Visually Grounded Text in Comics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![W&B: Tracked](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai)
[![Venue: JNLP Draft](https://img.shields.io/badge/Target-Journal%20of%20NLP-purple.svg)](https://www.cambridge.org/core/journals/natural-language-processing)

Implementation repository and experimental evaluation suite for the paper:  
**"Where does the image matter? Translating and evaluating visually grounded text in comics"**  
*Ravi Prakash Srivastava and Marie Escribe*  
*Target Venue: Journal of Natural Language Processing (Cambridge University Press)*

---

## 📌 1. Overview & Research Objectives

While recent studies demonstrate that visual context improves machine translation of comics and manga when scored by standard automatic metrics, critical open questions remain:
- **Where** does the image actually help, and **which** linguistic phenomena (e.g., gender, address forms, spatial deixis) strictly require visual grounding?
- Do Vision-Language Models (VLMs) **truly attend** to visual cues, or do they act invariantly even under corrupted or mismatched images?
- Can standard reference-free metrics (like COMET-Kiwi) and LLM-as-a-judge detect **image-dependent translation errors**?

### Research Questions (RQs)
- **RQ1 (Image Use & Grounding Veracity):** Does access to visual context measurably improve translation, and do models actively ground their decisions in the image? Tested via four visual conditions (*Text-Only*, *Correct Image*, *Mismatched Image*, *Noise/Blank*).
- **RQ2 (Linguistic Phenomena):** Which specific text–image relations and linguistic phenomena (gender disambiguation, honorifics/address forms, deixis, number, onomatopoeia, visual puns) benefit from, or are misled by, visual context?
- **RQ3 (Pipeline Error Breakdown):** How do cascaded pipelines (`Detection` $\to$ `OCR` $\to$ `MT`) compare with end-to-end multimodal VLMs, and where do translation errors originate (OCR noise vs. segmentation vs. semantic transfer)?
- **RQ4 (Evaluation Diagnostic Power):** Can automated metrics (COMET-Kiwi, chrF) and LLM judges spot image-dependent errors when benchmarked against human Multidimensional Quality Metrics (MQM)?

---

## 🔬 2. Benchmark Models for SOTA Comparison

To guarantee undeniable experimental authority and scientific rigor, the benchmark evaluates systems across **three distinct model tiers**, supported by text baselines and OCR engines:

| Tier / Category | Candidate Models | Key Capabilities & Justification |
| :--- | :--- | :--- |
| **Top Open VLM Benchmark** | **`Qwen2.5-VL-7B-Instruct`** *(or 72B if compute permits)* | The current undisputed leader in open vision-language benchmarks; superior document/comic OCR, reading order understanding, and fine-grained visual scene grounding. |
| **Alternative Open VLM** | **`Llama-3.2-11B-Vision-Instruct`** | Meta's flagship multimodal model; serves as an essential cross-architecture point of comparison against Qwen. |
| **Proprietary Upper Bound** | **`GPT-4o`** *(or Gemini 1.5 Pro / Claude 3.5 Sonnet)* | Represents the commercial ceiling of multimodal reasoning and natural conversational translation. |
| **Text-Only MT Baselines** | **`NLLB-200-3.3B`** (Meta) & **`Llama-3.1-8B-Instruct`** | Isolates whether vision-language models genuinely outperform specialized text translation engines on dialogue alone. |
| **Cascaded OCR Engines** | **`Manga-OCR`** & **`PaddleOCR`** | Measures error propagation and performance degradation caused by historical comic typefaces and stylized fonts. |

---

## 🧪 3. Experimental Matrix

### A. The 4 Visual Input Conditions (Adversarial Testing)
To probe whether models genuinely utilize visual information or hallucinate, every dialogue unit is evaluated under:
1. **$C_1$ (Text-Only):** Dialogue transcript provided with zero visual context.
2. **$C_2$ (Text + Correct Image):** Dialogue paired with its authentic comic panel crop.
3. **$C_3$ (Text + Mismatched Image):** Dialogue paired with an adversarial random panel from an unrelated comic scene (tests whether the model is misled by false visual context or ignores the image).
4. **$C_4$ (Text + Noise / Blank Image):** Dialogue paired with isotropic Gaussian noise or solid neutral gray image.

### B. Pipelines Evaluated
```
1. Cascaded Pipeline:
   Comic Panel ──▶ Bubble Detection ──▶ Manga-OCR / PaddleOCR ──▶ NLLB-200 / Llama-3.1 MT ──▶ Target Translation

2. Oracle Cascaded Pipeline:
   Gold Transcript (Human transcribed) ──▶ NLLB-200 / Llama-3.1 MT ──▶ Target Translation

3. End-to-End Multimodal VLM:
   [Comic Panel Image + Gold Transcript Prompt] ──▶ Qwen2.5-VL / Llama-Vision / GPT-4o ──▶ Target Translation
```

---

## 📊 4. Evaluation Suite & Human MQM Annotation

We implement a multi-tiered evaluation protocol:

1. **Human MQM (Multidimensional Quality Metrics):**
   - Subsets of 100–150 units per condition evaluated according to standard MQM guidelines (*Accuracy*, *Fluency*, *Terminology*, *Style*).
   - **Custom Tag:** `[IMAGE-DEPENDENT-ERROR]` specifically marks errors caused by missing or misinterpreting visual cues (e.g., misgendering a character clearly depicted as female, or misinterpreting a pointing gesture).
   - Second annotator overlap on 20% of data to report Cohen's $\kappa$ inter-annotator agreement.
2. **Reference-Free Quality Estimation (QE):**
   - `COMET-Kiwi` (`wmt22-cometkiwi-da` and `wmt23`).
3. **Reference-Based Metrics (on reference test subset):**
   - `chrF++`, `BLEU`, `COMET` (`wmt22-comet-da`), and `MetricX-24`.
4. **LLM-as-a-Judge Diagnostic Power:**
   - Multimodal VLM judge vs. Text-only LLM judge prompt-engineered with MQM severity criteria to measure if automatic judges correlate with human `[IMAGE-DEPENDENT-ERROR]` tags.
5. **Statistical Significance:**
   - Paired bootstrap resampling ($B = 10,000$) with Holm–Bonferroni correction.
   - Linear Mixed-Effects Model (`lmer`) modeling error severity with condition as fixed effect and item/model as random effects.

---

## ⚙️ 5. Hardware Requirements & Compute Environments

Execution requirements vary based on the deployment tier. For official paper results, **Option A** is the recommended configuration.

### 🌟 Option A: University GPU Cluster (HKBU GPU Cluster — Ideal for Paper Runs)
Running full-scale experiments across 300–500 dialogue units in unquantized `bfloat16` precision is best performed on institutional HPC infrastructure (such as the HKBU Faculty GPU Cluster).

- **Recommended Hardware Allocation:**
  - **GPU:** 1x NVIDIA A100 (40GB or 80GB VRAM) or 1x RTX 4090 (24GB VRAM)
  - **CPU:** 8 vCPUs / cores (for parallel panel cropping and batch tokenization)
  - **RAM:** 32 GB System Memory
  - **Storage:** 50 GB high-speed scratch space (for caching HuggingFace weights and comic crops)
- **Runtime Estimates (for 500 units $\times$ 4 visual conditions = 2,000 passes):**
  - `Qwen2.5-VL-7B-Instruct`: ~35 – 45 minutes on A100
  - `Llama-3.2-11B-Vision-Instruct`: ~50 – 70 minutes on A100
  - `COMET-Kiwi` Quality Estimation: ~5 – 10 minutes on A100
- **Automated SLURM Job Submission:**
  A ready-to-run cluster submission script is provided in [`scripts/run_cluster_slurm.sh`](scripts/run_cluster_slurm.sh):
  ```bash
  # Submit job to the cluster scheduler
  sbatch scripts/run_cluster_slurm.sh
  
  # Monitor job execution
  squeue -u $USER
  ```

### 💻 Option B: Local Consumer GPU (Prototyping & Pilot Runs)
If executing locally on consumer graphics cards:
- **Minimum VRAM:** 8 GB – 12 GB VRAM (RTX 3060, 3070, 4060, 4070).
- **Execution Mode:** Load models with 4-bit / 8-bit quantization (`bitsandbytes` or `AWQ`), or run via [Ollama](https://ollama.ai) (`ollama run qwen2.5-vl:7b`).

### ☁️ Option C: Zero-GPU Cloud API Mode
- Models like `GPT-4o`, `Gemini 1.5 Pro`, and `Claude 3.5 Sonnet` require **0 GB of local GPU memory**.
- System RAM requirements drop to standard laptop specs (2 GB – 4 GB RAM) as inference runs over HTTPS APIs.

### Hardware Summary Matrix
| Model / Pipeline Component | Precision / Mode | Minimum VRAM | Min System RAM | Local Disk |
| :--- | :--- | :--- | :--- | :--- |
| **`Qwen2.5-VL-7B-Instruct`** | Native `bfloat16` | **16 GB VRAM** | 16 GB | ~16 GB |
| **`Qwen2.5-VL-7B-Instruct`** | 4-bit Quantized | **6 – 8 GB VRAM** | 16 GB | ~5 GB |
| **`Llama-3.2-11B-Vision`** | Native `bfloat16` | **24 GB VRAM** | 32 GB | ~22 GB |
| **`Llama-3.2-11B-Vision`** | 4-bit Quantized | **8 – 10 GB VRAM** | 16 GB | ~7 GB |
| **`NLLB-200-3.3B` (Text MT)**| `fp16` | **7 GB VRAM** (or CPU) | 16 GB | ~7 GB |
| **`Manga-OCR` / `PaddleOCR`**| Standard | **2 – 4 GB VRAM** (or CPU) | 8 GB | ~1 GB |
| **`COMET-Kiwi` (Evaluation)** | Neural Metric | **4 – 6 GB VRAM** (or CPU) | 16 GB | ~3 GB |
| **`GPT-4o` / `Gemini 1.5 Pro`** | Cloud API | **0 GB GPU (None)** | 4 GB | Minimal |

---

## 📁 6. Repository Structure

```
comic-vlm-translation/
├── configs/
│   ├── models.yaml             # Model IDs, precision, temperature=0 settings
│   └── prompts.yaml            # Standardized prompts across all 4 visual conditions
│
├── data/
│   ├── raw/                    # Raw comic scans (Digital Comic Museum Golden Age)
│   ├── annotations/            # JSON annotations (bubble coordinates, gold text, labels)
│   │   └── curated_benchmark_500.json # Benchmark dataset with multimodal phenomena
│   └── crops/                  # Extracted panel crops and bubble regions
│
├── scripts/
│   ├── run_cluster_slurm.sh    # SLURM script for HKBU GPU Cluster (Option A)
│   └── clear_wandb.py          # Utility to clear dashboard runs
│
├── src/
│   ├── dataset/
│   │   ├── annotation_schema.py # Pydantic data schemas for units and relations
│   │   └── dcm_extractor.py     # Comic CBZ unpacker & panel slicer
│   │
│   ├── pipelines/
│   │   ├── cascaded.py          # OCR -> MT pipeline runner
│   │   ├── oracle_cascaded.py   # Gold text -> MT runner
│   │   └── vlm_end_to_end.py    # Multi-condition VLM inference runner
│   │
│   ├── evaluation/
│   │   ├── compute_metrics.py   # COMET-Kiwi, chrF, BLEU evaluation & bootstrap stats
│   │   ├── llm_judge.py         # Text & Multimodal LLM MQM-style evaluator
│   │   └── stats_analysis.py    # Bootstrap significance & mixed-effects models
│   │
│   └── utils/
│       ├── image_transforms.py  # Noise generator & mismatched image sampler
│       └── mqm_scorer.py        # Severity penalty calculator and aggregator
│
├── run_benchmark.py            # Primary 1-command benchmark runner (computes Tables 1-4 & W&B sync)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 7. Quickstart: Running on HPC & Fetching Live W&B Results

### A. Environment Setup
```bash
git clone https://github.com/er-ravi-srivastava/comic-vlm-translation.git
cd comic-vlm-translation

python -m venv venv
# Linux / Cluster:
source venv/bin/activate
# Windows PowerShell:
venv\Scripts\activate

pip install -r requirements.txt
```

### B. One-Command Publication Benchmark Run
To compute all four research questions and immediately stream publication-grade tables and charts to your **Weights & Biases** live dashboard:

```bash
# Authenticate W&B once (if not already set in environment)
wandb login

# Execute benchmark and live sync key paper metrics
python run_benchmark.py --wandb
```

#### Running on University GPU Cluster via SLURM:
```bash
# Submit full job to the HPC scheduler
sbatch scripts/run_cluster_slurm.sh

# Monitor status
squeue -u $USER
```

---

### 📈 Key Paper Metrics Fetched & Logged to W&B:

When you or any collaborator runs `python run_benchmark.py --wandb`, the following publication metrics are rendered in console and automatically logged to your W&B dashboard:

1. **Table 1: RQ1 - Impact of Visual Context (Grounding & Adversarial Test):**
   - Quality Estimation (`COMET-Kiwi`) & lexical accuracy (`chrF++`) across all 4 conditions:
     - $C_1$: Text-Only
     - $C_2$: Text + Correct Image
     - $C_3$: Text + Mismatched Image (Adversarial)
     - $C_4$: Text + Gaussian Noise / Blank
   - Observed effect size ($\Delta_{C_2 - C_1}$) and paired bootstrap statistical significance ($p$-value).
   - Adversarial degradation drop ($\Delta_{C_2 - C_3}$) proving true visual grounding.
2. **Table 2: RQ2 - Phenomenon-Level Breakdown:**
   - Accuracy and $\Delta$ gains over text-only across specific linguistic phenomena:
     - *Gender Disambiguation* (unexpressed English pronouns resolved by visual cues)
     - *Forms of Address* (*Tú/Usted* or *Tu/Vous* based on social hierarchy and uniforms)
     - *Spatial Deixis* ("Look at this / that")
     - *Number Agreement* (singular vs plural)
     - *Onomatopoeia & Sound FX*
3. **Table 3: RQ3 - Pipeline Breakdown & Error Attribution:**
   - Comparison across:
     - Pipeline A: Cascaded Real-World (`Bubble Detection` $\to$ `OCR` $\to$ `MT`)
     - Pipeline B: Oracle Baseline (`Gold Text` $\to$ `MT`)
     - Pipeline C: End-to-End Multimodal VLM (`Qwen2.5-VL`)
   - Quantifies the performance drop caused by historical comic typefaces (OCR Character Error Rate).
4. **Table 4: RQ4 - Evaluation Diagnostic Power & Blind Spot:**
   - Detection rates of `[IMAGE-DEPENDENT-ERROR]` across:
     - Human MQM (Gold standard)
     - Multimodal VLM Judge (with visual panel access)
     - Text-Only LLM Judge (demonstrates the evaluation blind spot)
     - Automated Metric (`COMET-Kiwi`)

---

### C. Running Individual Pipeline Components Manually (Optional)
```bash
# 1. Login to Weights & Biases (optional for online dashboard tracking)
wandb login

# 2. Run End-to-End VLM across all 4 visual conditions with W&B logging
python -m src.pipelines.vlm_end_to_end \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --data data/annotations/pilot_samples.json \
    --conditions text_only correct_image mismatched_image noise_image \
    --target_lang es \
    --output_dir outputs/qwen2.5_vl_7b/ \
    --wandb

# 3. Run Cascaded Baseline (OCR -> MT)
python -m src.pipelines.cascaded \
    --ocr paddleocr \
    --mt meta-llama/Llama-3.1-8B-Instruct \
    --data data/annotations/pilot_samples.json \
    --output_dir outputs/cascaded/
```

### Running Evaluation & LLM Judge (with W&B metrics sync)
```bash
# Compute COMET-Kiwi, chrF, and bootstrap significance tests (logged to W&B)
python -m src.evaluation.compute_metrics \
    --hypotheses outputs/qwen2.5_vl_7b/predictions.json \
    --gold data/annotations/pilot_samples.json \
    --wandb

# Run Multimodal LLM-as-a-Judge for Image-Dependent Errors
python -m src.evaluation.llm_judge \
    --judge gpt-4o \
    --hypotheses outputs/qwen2.5_vl_7b/predictions.json \
    --mode multimodal
```

---

## 📜 Declarations & Ethics
- **Data Source:** Public domain Golden Age comic books from the Digital Comic Museum (c. 1938–1956), verified for non-renewal under US copyright law. Titles overlapping with the CoMix benchmark are systematically filtered out.
- **Human Annotation:** Conducted under strict MQM guidelines with explicit tracking of multimodal ambiguity.

---

## 📚 References
- Baek, J. et al. (2026). MangaVQA and MangaLMM: A benchmark and specialized model for multimodal manga understanding. *Findings of EACL 2026*.
- Elliott, D. (2018). Adversarial evaluation of multimodal machine translation. *EMNLP 2018*.
- Futeral, M. et al. (2023). Tackling ambiguity with images: Improved multimodal machine translation. *ACL 2023*.
- Lippmann, P. et al. (2025). Context-informed machine translation of manga using multimodal large language models. *COLING 2025*.
- Lommel, A. et al. (2014). Multidimensional Quality Metrics (MQM): A framework for declaring translation quality. *Revista Tradumàtica*.
- Martinec, R., & Salway, A. (2005). A system for image–text relations in new (and old) media. *Visual Communication*.
- Vivoli, E. et al. (2024). CoMix: A comprehensive benchmark for multi-task comic understanding. *NeurIPS 2024*.
- Yirmibeşoğlu Balal, Z. & Güngör, T. (2026). Lexical diversity and translation literalness in machine translation. *Natural Language Processing (JNLP)*.
