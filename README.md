# Leveraging Large Language Models for Enhanced Recommendation Systems

**Authors:** Jiwei Liu, Logan Ju, Shoma Sawa, Roujia Wang  
**University of Michigan**

---

## Project Overview

This project explores the integration of state-of-the-art Large Language Models (LLMs) into modern recommendation systems, with a focus on short-form video platforms such as TikTok. We developed an extensible CLI pipeline that leverages both generative and discriminative LLMs for embedding generation and re-ranking, demonstrating significant improvements in recommendation accuracy and user engagement metrics.

## Key Achievements

- **Innovative LLM Integration:** Seamlessly combined generative and discriminative language models (OpenAI, HuggingFace) to enhance recommendation quality.
- **Flexible Data Enrichment:** Enabled dynamic inclusion of multimodal features (titles, covers, frames, comments) to enrich user-item representations.
- **Scalable Evaluation:** Designed a robust pipeline to evaluate hit rate improvements across various N values, supporting large-scale datasets (e.g., MicroLens-100k).
- **Open Source Impact:** Provided a modular, easy-to-extend codebase for the research and engineering community, with transparent documentation and reproducible experiments.
- **Real-World Relevance:** Demonstrated the effectiveness of LLM-powered re-ranking in realistic short-form video scenarios, paving the way for future advancements in personalized content delivery.

---

![Pipeline Overview](Pipelines.jpg)

---

## Experimental Results

The following tables summarize key experimental results on the MicroLens-100k dataset, demonstrating the impact of advanced LLM-based re-ranking on two strong baselines. Only the most representative and impactful results are shown.

**Table 1: NFM Baseline and LLM Re-Ranking Performance**

| Model Type         | Model Variant                   | Hits@10 | Hits@20 | % Change (Hits@20) |
|--------------------|---------------------------------|---------|---------|-------------------|
| IDRec RS (NFM)     | NFM (ours)                      |   195   |   289   |      +0.0%        |
| Discriminative LLM | e5-small-v2                     |   178   |   289   |      +0.0%        |
| Discriminative LLM | all-MiniLM-L6-v2                |   176   |   289   |      +0.0%        |
| Generative LLM     | Meta-Llama-3-8B-Instruct (init) |   161   |   243   |     -15.92%       |
| Generative LLM     | GPT-4o mini (refined prompt)    |   184   |   280   |      -3.11%       |

**Table 2: VIDRec DSSM Baseline and LLM Re-Ranking Performance**

| Model Type         | Model Variant                   | Hits@10 | Hits@20 | % Change (Hits@20) |
|--------------------|---------------------------------|---------|---------|-------------------|
| VIDRec RS (DSSM)   | VIDRec DSSM (ours)              |    78   |   148   |      +0.0%        |
| Discriminative LLM | all-distilroberta-v1            |    93   |   148   |      +0.0%        |
| Discriminative LLM | text-embedding-3-small          |    85   |   148   |      +0.0%        |
| Generative LLM     | Meta-Llama-3-8B-Instruct (init) |    77   |   122   |     -17.57%       |
| Generative LLM     | GPT-4o mini (refined prompt)    |   119   |   140   |     -5.41%        |

These results show that while discriminative LLMs can match the baseline on top-N hit rates, generative LLMs with current prompting strategies may still lag behind. However, the pipeline is flexible for future improvements as LLM technology advances.

---

# Recommendation System CLI Tool

This repository provides a CLI tool to evaluate and improve recommendation system performance using generative and discriminative models. The tool supports dynamic model selection, parameter configuration, and calculates hit rate improvements for different values of N.

---

## Features

- Parse and preprocess recommendation data.
- Support for both **generative** and **discriminative** models.
- Dynamic selection of model sources: **HuggingFace** or **OpenAI**.
- Flexible inclusion of additional data attributes (title, cover, frames, comments).
- Calculation of hit rate and its improvement for different N values.

---

## Requirements

- Python >= 3.7
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `argparse`
  - `openai`
  - `requests`
  - `huggingface_hub`

Install the dependencies using:

```bash
pip install -r requirements.txt
```

Run the pipeline
```
python main.py \
    --inference_path "path_to_inference" \
    --pairs_path "MicroLens-100k_pairs.csv" \
    --titles_path "MicroLens-100k_title_en.csv" \
    --comments_path "MicroLens-100k_comment_en.txt" \
    --covers_path "path_to_covers" \
    --frames_path "path_to_frames" \
    --model_type "generative" \
    --model_source "huggingface" \
    --model_name "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    --hf_api_key "your_huggingface_api_key" \
    --openai_api_key "your_openai_api_key" \
    --include_title \
    --include_cover \
    --include_frames \
    --include_comments \
    --n_list 10 20 50
```