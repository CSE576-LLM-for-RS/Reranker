import argparse
from data import Data
from generative import Generative_Model
from discriminative import Discriminative_Model
from eval import Evaluation


def main():
    # Parse command-line arguments
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
        default=[10, 20, 50],
        help="List of N values for hit rate",
    )

    args = parser.parse_args()

    # Load data
    data = Data(
        pairs_path=args.pairs_path,
        titles_path=args.titles_path,
        comments_path=args.comments_path,
        covers_path=args.covers_path,
        frames_path=args.frames_path,
    )

    model = None
    if args.model_type == "generative":
        model = Generative_Model(
            name=args.model_name,
            source=args.model_source,
            data=data,
            hf_api_token=args.hf_api_key,
            openai_api_key=args.openai_api_key,
        )
    elif args.model_type == "discriminative":
        model = Discriminative_Model(
            name=args.model_name,
            source=args.model_source,
            data=data,
            hf_api_key=args.hf_api_key,
            openai_api_key=args.openai_api_key,
        )

    evaluation = Evaluation(data)
    original_recommendations = evaluation.parse_recommendations(args.inference_path)

    new_recommendations = {}
    for user_id, items in original_recommendations.items():
        if args.model_type == "generative":
            if args.model_source == "huggingface":
                new_recommendations[user_id] = model.reranker_llama_stream(
                    user_id,
                    items,
                    model_name=args.model_name,
                    include_title=args.include_title,
                    include_cover=args.include_cover,
                    include_frames=args.include_frames,
                    include_comments=args.include_comments,
                )
            elif args.model_source == "openai":
                new_recommendations[user_id] = model.reranker_gpt(
                    user_id,
                    items,
                    model_name=args.model_name,
                    include_title=args.include_title,
                    include_cover=args.include_cover,
                    include_frames=args.include_frames,
                    include_comments=args.include_comments,
                )
        elif args.model_type == "discriminative":
            if args.model_source == "huggingface":
                new_recommendations[user_id] = model.reranker_sentence_transformers(
                    user_id,
                    items,
                    model_name=args.model_name,
                    include_title=args.include_title,
                    include_cover=args.include_cover,
                    include_frames=args.include_frames,
                    include_comments=args.include_comments,
                )
            elif args.model_source == "openai":
                new_recommendations[user_id] = model.reranker_text_embedding_model(
                    user_id,
                    items,
                    model_name=args.model_name,
                    include_title=args.include_title,
                    include_cover=args.include_cover,
                    include_frames=args.include_frames,
                    include_comments=args.include_comments,
                )

    improvements = {}
    for N in args.n_list:
        baseline_hit_rate = evaluation.evaluate_hit_rate(original_recommendations, N)
        new_hit_rate = evaluation.evaluate_hit_rate(new_recommendations, N)

        improvement = (
            ((new_hit_rate - baseline_hit_rate) / baseline_hit_rate * 100)
            if baseline_hit_rate > 0
            else float("inf")
        )
        improvements[N] = {
            "baseline_hit_rate": baseline_hit_rate,
            "new_hit_rate": new_hit_rate,
            "improvement": improvement,
        }

    for N, result in improvements.items():
        print(f"\nFor N = {N}:")
        print(f"  Baseline Hit Rate: {result['baseline_hit_rate']:.4f}")
        print(f"  New Hit Rate: {result['new_hit_rate']:.4f}")
        print(f"  Hit Rate Improvement: {result['improvement']:.2f}%")


if __name__ == "__main__":
    main()
