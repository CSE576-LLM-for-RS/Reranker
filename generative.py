import requests
from typing import List
from data import Data


class Discriminative_Model:
    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source

    def reranker_sentence_transformers(
        user_id: int,
        item_list: List[int],
        model_name: str,
        include_title=True,
        include_cover=False,
        include_frames=False,
        include_comments=False,
    ) -> List[int]:
        previous_item_info = ""
        previous_item_list = get_N_latest_items(user_id, 10)
        for item in previous_item_list:
            item_title = extract_title(item)
            previous_item_info += f"{item_title}.\n"

        item_list = filter_items(item_list)

        items_info = {item: "" for item in item_list}
        if include_title:
            titles_dict = {item: extract_title(item) for item in item_list}
            for item, title in titles_dict.items():
                items_info[item] += f"Title: {title}\n"

        if include_cover:
            covers_dict = {item: process_cover(item, covers_path) for item in item_list}
            for item, cover in covers_dict.items():
                items_info[item] += f"Cover: {cover}\n"

        if include_frames:
            frames_dict = {
                item: process_frames(item, frames_path) for item in item_list
            }
            for item, frames in frames_dict.items():
                for i, frame in enumerate(frames, start=1):
                    items_info[item] += f"Frame_{i}: {frame}\n"

        if include_comments:
            comments_dict = {
                item: extract_comments(user_id, item) for item in item_list
            }
            for item, comments in comments_dict.items():
                for i, comment in enumerate(comments, start=1):
                    items_info[item] += f"Comment_{i}: {comment}\n"

        payload = {
            "inputs": {
                "source_sentence": previous_item_info,
                "sentences": [items_info[item] for item in item_list],
            }
        }
        result = requests.post(
            f"https://api-inference.huggingface.co/models/{model_name}",
            headers=headers,
            json=payload,
        ).json()
        print(f"API response: {result}")
        # print(result)
        # result is a list of scores for each item in the item_list
        # return id of the ranked items from highest score to lowest score
        ranked_items = [item_list[i] for i in np.argsort(result)]
        for x in ranked_items:
            if x not in item_list:
                print("1")
        return ranked_items[::-1]
