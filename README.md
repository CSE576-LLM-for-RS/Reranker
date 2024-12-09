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
