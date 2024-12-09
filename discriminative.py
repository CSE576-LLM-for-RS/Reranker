import numpy as np
from typing import List
from data import Data
import openai
import requests


class Discriminative_Model:
    def __init__(
        self, name: str, source: str, data: Data, hf_api_key: str, openai_api_key: str
    ):
        self.name = name
        self.source = source
        self.data = data
        self.headers = {"Authorization": f"Bearer {hf_api_key}"}
        self.openai_client = openai.Client(api_key=openai_api_key)

    def reranker_sentence_transformers(
        self,
        user_id: int,
        item_list: List[int],
        model_name: str,
        include_title=True,
        include_cover=False,
        include_frames=False,
        include_comments=False,
    ) -> List[int]:
        previous_item_info = ""
        previous_item_list = self.data.get_N_latest_items(user_id, 10)

        # Compile previous item information
        for item in previous_item_list:
            item_title = self.data.extract_title(item)
            previous_item_info += f"{item_title}.\n"

        # Filter valid items
        item_list = self.data.filter_items(item_list)

        # Prepare information for items
        items_info = {item: "" for item in item_list}

        # Add title information
        if include_title:
            for item in item_list:
                title = self.data.extract_title(item)
                items_info[item] += f"Title: {title}\n"

        # Add cover information
        if include_cover and self.data.covers_path:
            for item in item_list:
                cover_text = self.data.process_cover(item)
                if cover_text:
                    items_info[item] += f"Cover: {cover_text}\n"

        # Add frames information
        if include_frames and self.data.frames_path:
            for item in item_list:
                frames_text = self.data.process_frames(item)
                for i, frame in enumerate(frames_text, start=1):
                    items_info[item] += f"Frame_{i}: {frame}\n"

        # Add comments information
        if include_comments and self.data.comments is not None:
            for item in item_list:
                comments = self.data.extract_comments(user_id, item)
                for i, comment in enumerate(comments, start=1):
                    items_info[item] += f"Comment_{i}: {comment}\n"

        # Prepare payload for the API
        payload = {
            "inputs": {
                "source_sentence": previous_item_info,
                "sentences": [items_info[item] for item in item_list],
            }
        }

        # Send request to model API
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_name}",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Validate and process result
            if isinstance(result, list) and all(
                isinstance(score, (int, float)) for score in result
            ):
                ranked_items = [item_list[i] for i in np.argsort(result)]
                return ranked_items[::-1]
            else:
                raise ValueError(f"Unexpected response format: {result}")

        except requests.RequestException as e:
            raise ConnectionError(f"Error with model API request: {e}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

    def reranker_text_embedding_model(
        self,
        user_id: int,
        item_list: List[int],
        model_name: str,
        include_title=True,
        include_cover=False,
        include_frames=False,
        include_comments=False,
    ) -> List[int]:
        # Get previous item information
        previous_item_list = self.data.get_N_latest_items(user_id, 10)
        total_item_info = ""

        for item in previous_item_list:
            item_title = self.data.extract_title(item)
            total_item_info += item_title

        # Calculate average embedding for previous items
        agg_previous_item_embedding = self.openai_client.Embedding.create(
            input=[total_item_info], model=model_name
        )["data"][0]["embedding"]

        # Filter valid items
        item_list = self.data.filter_items(item_list)

        # Prepare item information
        items_info = {item: "" for item in item_list}

        if include_title:
            for item in item_list:
                title = self.data.extract_title(item)
                items_info[item] += f"Title: {title}\n"

        if include_cover and self.data.covers_path:
            for item in item_list:
                cover_text = self.data.process_cover(item)
                if cover_text:
                    items_info[item] += f"Cover: {cover_text}\n"

        if include_frames and self.data.frames_path:
            for item in item_list:
                frames_text = self.data.process_frames(item)
                for i, frame in enumerate(frames_text, start=1):
                    items_info[item] += f"Frame_{i}: {frame}\n"

        if include_comments and self.data.comments is not None:
            for item in item_list:
                comments = self.data.extract_comments(user_id, item)
                for i, comment in enumerate(comments, start=1):
                    items_info[item] += f"Comment_{i}: {comment}\n"

        # Calculate embeddings for items
        item_embeddings = {}
        for item in item_list:
            embedding = self.openai_client.Embedding.create(
                input=[items_info[item]], model=model_name
            )["data"][0]["embedding"]
            item_embeddings[item] = embedding

        # Calculate cosine similarity between average embedding and items
        scores = {
            item: np.dot(agg_previous_item_embedding, item_embeddings[item])
            / (
                np.linalg.norm(agg_previous_item_embedding)
                * np.linalg.norm(item_embeddings[item])
            )
            for item in item_list
        }

        # Rank items by similarity score
        ranked_items = [
            item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return ranked_items
