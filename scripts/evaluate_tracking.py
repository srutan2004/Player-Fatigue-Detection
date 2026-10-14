"""Run the separate held-out tracker audit; no identity hints or training."""

from src.evaluate import evaluate_mot

if __name__ == "__main__":
    print(evaluate_mot())
