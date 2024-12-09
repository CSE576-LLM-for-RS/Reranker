import os
import re
import pandas as pd
import requests
from typing import List, Dict


class Data:
    def __init__(
        self,
        pairs_path: str,
        titles_path: str,
        comments_path: str = None,
        covers_path: str = None,
        frames_path: str = None,
    ):
        self.pairs = pd.read_csv(pairs_path)
        self.titles = pd.read_csv(titles_path, header=None)
        self.titles.columns = ["item", "title"]
        self.comments = None
        self.covers_path = covers_path
        self.frames_path = frames_path

        if comments_path:
            self.comments = pd.read_csv(
                comments_path,
                sep="\t",
                header=None,
                names=["user_id", "item_id", "comment"],
                dtype={"user_id": int, "item_id": int, "comment": str},
            )

    def parse_recommendations(self, file_path: str) -> Dict[int, List[int]]:
        try:
            with open(file_path, "r") as file:
                content = file.read()

            pattern = re.compile(r"User (\d+): Top 20 recommended items: \[([^\]]+)\]")
            data = pattern.findall(content)

            recommendation_dict = {
                int(user): list(map(int, items.split(", "))) for user, items in data
            }
            return recommendation_dict

        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_valid_users(self) -> List[int]:
        return self.pairs["user"].unique().tolist()

    def get_valid_items(self) -> List[int]:
        return self.pairs["item"].unique().tolist()

    def get_all_users(self) -> List[int]:
        return self.get_valid_users()

    def get_all_items(self) -> List[int]:
        return self.get_valid_items()

    def is_valid_user(self, user_id: int) -> bool:
        return user_id in self.get_all_users()

    def is_valid_item(self, item_id: int) -> bool:
        return item_id in self.get_all_items()

    def get_true_item(self, user_id: int) -> int:
        user_items = self.pairs[self.pairs["user"] == user_id]
        if user_items.empty:
            raise ValueError(f"No items found for user {user_id}.")

        true_item = user_items.sort_values(by="timestamp", ascending=False)[
            "item"
        ].iloc[0]
        return true_item

    def get_N_latest_items(self, user_id: int, N: int) -> List[int]:
        user_items = self.pairs[self.pairs["user"] == user_id]
        if user_items.empty:
            raise ValueError(f"No items found for user {user_id}.")

        latest_items = (
            user_items.sort_values(by="timestamp", ascending=False)["item"]
            .head(N + 2)
            .tolist()
        )
        return latest_items[2:]

    def extract_comments(self, user_id: int, item_id: int) -> List[str]:
        if self.comments is None:
            raise ValueError("Comments data is not loaded.")

        comments_list = self.comments[
            (self.comments["user_id"] == user_id)
            & (self.comments["item_id"] == item_id)
        ]["comment"]
        if comments_list.empty:
            return []
        return comments_list.tolist()

    def extract_title(self, item_id: int) -> str:
        if item_id <= 0:
            raise ValueError(f"Item ID {item_id} is not valid.")

        title_row = self.titles[self.titles["item"] == item_id]
        if title_row.empty:
            raise ValueError(f"No title found for Item ID {item_id}.")
        return title_row["title"].iloc[0]

    def image2text(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            data = f.read()
        response = requests.post(Image_model_API_URL, headers=headers, data=data)
        response_data = response.json()

        if not response_data or "generated_text" not in response_data[0]:
            raise ValueError(
                f"Invalid response from image processing API for {image_path}."
            )
        return response_data[0]["generated_text"]

    def process_cover(self, item_id: int) -> str:
        if not self.covers_path:
            raise ValueError("Covers folder path is not provided.")

        file_name = f"{item_id}.jpg"
        file_path = os.path.join(self.covers_path, file_name)
        if os.path.exists(file_path):
            try:
                return self.image2text(file_path)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File {file_name} not found in {self.covers_path}")

    def process_frames(self, item_id: int) -> List[str]:
        if not self.frames_path:
            raise ValueError("Frames folder path is not provided.")

        text_list = []
        for i in range(1, 6):
            file_name = f"{item_id}-{i}.jpg"
            file_path = os.path.join(self.frames_path, file_name)

            if os.path.exists(file_path):
                try:
                    text = self.image2text(file_path)
                    text_list.append(text)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
            else:
                print(f"File {file_name} not found in {self.frames_path}")

        if not text_list:
            print(f"No valid text extracted for item_id {item_id}.")
        return text_list

    def filter_items(self, item_list: List[int]) -> List[int]:
        return [item for item in item_list if self.is_valid_item(item)]

    def hit_rate(self, user_id_list: List[int], item_id_list: List[List[int]]) -> float:
        hits = 0
        for user_id, items in zip(user_id_list, item_id_list):
            items = self.filter_items(items)
            if not items:
                continue
            if self.get_true_item(user_id) in items:
                hits += 1
        return hits / len(user_id_list)

    def ndcg(self, user_id_list: List[int], item_id_list: List[List[int]]) -> float:
        # NDCG calculation placeholder
        pass
