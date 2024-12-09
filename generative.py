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
