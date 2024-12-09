from typing import List, Dict
from data import Data


class Evaluation:
    def __init__(self, data: Data):
        self.data = data

    def evaluate_hit_rate(
        self, user_recommendations: Dict[int, List[int]], N: int
    ) -> float:
        user_ids = list(user_recommendations.keys())
        recommendations = list(user_recommendations.values())

        hit_rate = self.data.hit_rate(user_ids, recommendations[:N])
        return hit_rate
