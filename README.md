# Where Does the Image Matter? Translating and Evaluating Visually Grounded Text in Comics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
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

## 📁 5. Repository Structure

```
comic-vlm-translation/
├── configs/
│   ├── models.yaml             # Model IDs, API keys, temperature=0 parameters
│   └── prompts.yaml            # Standard prompts for all 4 visual conditions & LLM judges
│
├── data/
│   ├── raw/                    # Raw comic scans (Digital Comic Museum Golden Age)
│   ├── annotations/            # JSON annotations (bubble coordinates, gold text, labels)
│   └── crops/                  # Extracted panel crops and bubble regions
│
├── src/
│   ├── dataset/
│   │   ├── annotation_schema.py # Pydantic data schemas for units and relations
│   │   └── dcm_extractor.py     # Comic CBZ unpacker & panel slicer
│   │
│   ├── pipelines/
│   │   ├── cascaded.py          # OCR -> MT pipeline runner
│   │   ├── oracle_cascaded.py   # Gold text -> MT runner
│   │   └── vlm_end_to_end.py    # Qwen2.5-VL / Llama-3.2-Vision / GPT-4o runner
│   │
│   ├── evaluation/
│   │   ├── compute_metrics.py   # COMET-Kiwi, chrF, BLEU evaluation
│   │   ├── llm_judge.py         # Text & Multimodal LLM MQM-style evaluator
│   │   └── stats_analysis.py    # Bootstrap significance & mixed-effects models
│   │
│   └── utils/
│       ├── image_transforms.py  # Noise generator & mismatched image sampler
│       └── mqm_scorer.py        # Severity penalty calculator and aggregator
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 6. Quickstart & Installation

### Environment Setup
```bash
git clone https://github.com/er-ravi-srivastava/comic-vlm-translation.git
cd comic-vlm-translation

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Running Inference across the 4 Visual Conditions
```bash
# Run End-to-End VLM across all 4 visual conditions
python -m src.pipelines.vlm_end_to_end \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --data data/annotations/pilot_samples.json \
    --conditions text_only correct_image mismatched_image noise_image \
    --target_lang es \
    --output_dir outputs/qwen2.5_vl_7b/

# Run Cascaded Baseline (OCR -> MT)
python -m src.pipelines.cascaded \
    --ocr paddleocr \
    --mt meta-llama/Llama-3.1-8B-Instruct \
    --data data/annotations/pilot_samples.json \
    --output_dir outputs/cascaded/
```

### Running Evaluation & LLM Judge
```bash
# Compute COMET-Kiwi and chrF
python -m src.evaluation.compute_metrics \
    --hypotheses outputs/qwen2.5_vl_7b/predictions.json \
    --gold data/annotations/pilot_samples.json

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
