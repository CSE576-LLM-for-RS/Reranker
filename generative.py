import re
from typing import List
from data import Data
from huggingface_hub import InferenceClient
import openai


class Generative_Model:
    def __init__(
        self,
        name: str,
        source: str,
        data: Data,
        hf_api_token: str = None,
        openai_api_key: str = None,
    ):
        self.name = name
        self.source = source
        self.data = data
        self.hf_client = InferenceClient(api_key=hf_api_token) if hf_api_token else None
        self.openai_client = openai if openai_api_key else None

    def reranker_llama_stream(
        self,
        user_id: int,
        item_list: List[int],
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        include_title=True,
        include_cover=False,
        include_frames=False,
        include_comments=False,
    ) -> List[int]:
        prompt = "You are an expert in recommending items to users.\n"
        previous_item_list = self.data.get_N_latest_items(user_id, 10)

        # Build prompt
        for item in previous_item_list:
            item_title = self.data.extract_title(item)
            prompt += f"User has previously interacted with this item with title: {item_title}.\n"

        prompt += f"These are the item_ids that you need to rerank: {item_list}.\n"
        prompt += "Below is the information of these items.\n"

        item_list = self.data.filter_items(item_list)

        if include_title:
            for item in item_list:
                title = self.data.extract_title(item)
                prompt += f"Item_Id: {item}. Title: {title}\n"

        if include_cover and self.data.covers_path:
            for item in item_list:
                cover = self.data.process_cover(item)
                prompt += f"Item_Id: {item}. Cover: {cover}\n"

        if include_frames and self.data.frames_path:
            for item in item_list:
                frames = self.data.process_frames(item)
                for i, frame in enumerate(frames, start=1):
                    prompt += f"Item_Id: {item}. Frame_{i}: {frame}\n"

        if include_comments and self.data.comments is not None:
            for item in item_list:
                comments = self.data.extract_comments(user_id, item)
                for i, comment in enumerate(comments, start=1):
                    prompt += f"Item_Id: {item}. Comment_{i}: {comment}\n"

        prompt += (
            "\nPlease rerank these items for the user based on their relevance.\n"
            "Make sure to include ALL given item IDs in your response, without omitting any.\n"
            "Output ONLY the reranked list of item IDs, separated by commas, in the exact order of ranking.\n"
            "Example output: 123, 456, 789\n"
        )

        # Stream response from Hugging Face Llama model
        stream = self.hf_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5000,
            stream=True,
        )

        result_string = ""
        for chunk in stream:
            result_string += chunk.choices[0].delta.content

        match = re.search(r"(?:\d+\s*,\s*)+\d+\b", result_string)
        if match:
            reranked_items = list(map(int, match.group().split(",")))
            return reranked_items
        else:
            print("Llama stream failed to produce a valid response.")
            return item_list

    def reranker_gpt(
        self,
        user_id: int,
        item_list: List[int],
        model_name: str,
        include_title=True,
        include_cover=False,
        include_frames=False,
        include_comments=False,
    ) -> List[int]:
        if not self.openai_client:
            raise ValueError("OpenAI API client is not configured.")

        prompt = "You are an expert in recommending items to users.\n"
        previous_item_list = self.data.get_N_latest_items(user_id, 10)

        # Build prompt
        for item in previous_item_list:
            item_title = self.data.extract_title(item)
            prompt += f"User {user_id} has previously interacted with this item with title: {item_title}.\n"

        prompt += f"These are the item_ids that you need to rerank: {item_list}.\n"
        prompt += "Below is the information of these items.\n"

        item_list = self.data.filter_items(item_list)

        if include_title:
            for item in item_list:
                title = self.data.extract_title(item)
                prompt += f"Item_Id: {item}. Title: {title}\n"

        if include_cover and self.data.covers_path:
            for item in item_list:
                cover = self.data.process_cover(item)
                prompt += f"Item_Id: {item}. Cover: {cover}\n"

        if include_frames and self.data.frames_path:
            for item in item_list:
                frames = self.data.process_frames(item)
                for i, frame in enumerate(frames, start=1):
                    prompt += f"Item_Id: {item}. Frame_{i}: {frame}\n"

        if include_comments and self.data.comments is not None:
            for item in item_list:
                comments = self.data.extract_comments(user_id, item)
                for i, comment in enumerate(comments, start=1):
                    prompt += f"Item_Id: {item}. Comment_{i}: {comment}\n"

        prompt += (
            "\nPlease rerank these items for the user based on their relevance.\n"
            "Make sure to include ALL given item IDs in your response, without omitting any.\n"
            "Output ONLY the reranked list of item IDs, separated by commas, in the exact order of ranking.\n"
            "Example output: 123, 456, 789\n"
        )

        response = self.openai_client.ChatCompletion.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
        )

        result_content = response.choices[0].message["content"]
        try:
            ranked_items = list(map(int, result_content.split(",")))
            return ranked_items
        except ValueError:
            print("GPT failed to produce a valid response.")
            return item_list
