from typing import List, Dict
from data import Data


class Evaluation:
    def __init__(self, data: Data):
        self.data = data

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
            return {}

    def evaluate_hit_rate(
        self, user_recommendations: Dict[int, List[int]], N: int
    ) -> float:
        user_ids = list(user_recommendations.keys())
        recommendations = list(user_recommendations.values())

        hit_rate = self.data.hit_rate(user_ids, recommendations[:N])
        return hit_rate

    def get_improvement(
        self,
        baseline_dict: Dict[int, List[int]],
        new_dict: Dict[int, List[int]],
        N_list: List[int],
    ) -> Dict[int, float]:
        baseline_hit_rates = {}

        new_hit_rates = {}

        # Calculate hit rates for each N
        for N in N_list:
            baseline_hit_rates[N] = self.calculate_hit_rate(baseline_dict, N)
            new_hit_rates[N] = self.calculate_hit_rate(new_dict, N)

        # Calculate improvement
        improvement = {}
        for N in N_list:
            baseline_hr = baseline_hit_rates.get(N, 0)
            new_hr = new_hit_rates.get(N, 0)

            if baseline_hr == 0:
                improvement[N] = float("inf")  # Avoid division by zero
            else:
                improvement[N] = ((new_hr - baseline_hr) / baseline_hr) * 100

        return improvement
