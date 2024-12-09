import argparse
from data import Data
from generative import Generative_Model
from discriminative import Discriminative_Model
from eval import Evaluation
from typing import Dict, List


def main():
    parser = argparse.ArgumentParser(description="Recommendation Model CLI")
    parser.add_argument(
        "--inference_path", required=True, help="Path to inference file"
    )
    parser.add_argument("--pairs_path", required=True, help="Path to pairs CSV file")
    parser.add_argument("--titles_path", required=True, help="Path to titles CSV file")
    parser.add_argument("--comments_path", help="Path to comments TXT file")
    parser.add_argument("--covers_path", help="Path to covers directory")
    parser.add_argument("--frames_path", help="Path to frames directory")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=["generative", "discriminative"],
        help="Model type",
    )
    parser.add_argument(
        "--model_source",
        required=True,
        choices=["huggingface", "openai"],
        help="Model source",
    )
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--hf_api_key", help="HuggingFace API key")
    parser.add_argument("--openai_api_key", help="OpenAI API key")
    parser.add_argument(
        "--include_title",
        action="store_true",
        default=True,
        help="Include title information",
    )
    parser.add_argument(
        "--include_cover", action="store_true", help="Include cover information"
    )
    parser.add_argument(
        "--include_frames", action="store_true", help="Include frames information"
    )
    parser.add_argument(
        "--include_comments", action="store_true", help="Include comments information"
    )
    parser.add_argument(
        "--n_list",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="List of N values for hit rate",
    )

    args = parser.parse_args()


if __name__ == "__main__":
    main()
