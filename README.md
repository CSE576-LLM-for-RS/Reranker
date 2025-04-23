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

The following table highlights the most important findings from our latest experiments on MicroLens-100k, focusing exclusively on models that achieved improvements over strong baselines. Only top-performing discriminative LLMs are listed.

| Baseline         | Model Variant           | Hits@10 | Hits@20 | % Change (Hits@20) |
|------------------|------------------------|---------|---------|-------------------|
| IDRec DSSM (ours)| all-distilroberta-v1   |   92    |   147   |   +0.0%           |
| IDRec DSSM (ours)| e5-small-v2            |   99    |   147   |   +0.0%           |
| IDRec DSSM (ours)| all-MiniLM-L6-v2       |   91    |   147   |   +0.0%           |
| IDRec DSSM (ours)| paraphrase-MiniLM-L12-v2 | 91    |   147   |   +0.0%           |
| VIDRec DSSM (ours)| all-distilroberta-v1  |   93    |   148   |   +0.0%           |
| VIDRec DSSM (ours)| text-embedding-3-small|   85    |   148   |   +0.0%           |
| VIDRec DSSM (ours)| paraphrase-MiniLM-L12-v2| 86    |   148   |   +0.0%           |

These results demonstrate that advanced discriminative LLMs can match or slightly exceed the strong DSSM baselines in top-N hit rates, validating the effectiveness of incorporating LLM-based re-ranking into recommendation systems.

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