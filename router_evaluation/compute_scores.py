# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import math

# Add parent directory to path to import ModelNameManager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_arena_score(cost, accuracy, beta=0.1, c_max=200, c_min=0.0044):
    """
    Compute the score S_i,β for a given cost and accuracy.

    Parameters:
    -----------
    cost : float
        The cost c_i of the model or router.
    accuracy : float
        The accuracy A_i of the model or router.
    beta : float, optional
        Weighting factor between accuracy and cost (default = 0.1).
    c_max : float, optional
        Maximum cost (default = 200).
    c_min : float, optional
        Minimum cost (default = 0.0044).

    Returns:
    --------
    float
        The computed score S_i,β.
    """

    # Compute normalized cost C_i
    C_i = (math.log2(c_max) - math.log2(cost)) / (math.log2(c_max) - math.log2(c_min))

    # Compute score S_i,β
    S = ((1 + beta) * accuracy * C_i) / (beta * accuracy + C_i)

    return S


def compute_scores(router_name: str):
    """
    Load router prediction file and compute accuracy, total cost, and arena score.

    Args:
        router_name: Name of the router (corresponds to prediction file name)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prediction_path = os.path.join(
        script_dir, "../router_inference/predictions", f"{router_name}.json"
    )

    # Check if file exists
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}\n"
            f"Please make sure the router name is correct and the prediction file exists."
        )

    # Load the prediction file
    print(f"Loading predictions from {prediction_path}...")
    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not predictions:
        print(f"Warning: No predictions found for router {router_name}")
        return None

    print(f"Loaded {len(predictions)} predictions\n")

    # Extract accuracy and cost for all queries
    accuracies = []
    costs = []
    valid_cost_count = 0

    for prediction in predictions:
        # Extract accuracy
        accuracy = prediction.get("accuracy")
        if accuracy is not None:
            accuracies.append(accuracy)

        # Extract cost
        cost = prediction.get("cost")
        if cost is not None and cost > 0:
            costs.append(cost)
            valid_cost_count += 1

    # Compute average accuracy
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    # Compute total cost (sum of all costs)
    total_cost = sum(costs) if costs else 0.0

    # Compute average cost per 1000 queries for arena score calculation
    num_queries = len(predictions)
    avg_cost_per_1000 = (total_cost / num_queries * 1000) if num_queries > 0 else 0.0

    # Compute arena score using average cost per 1000 queries and average accuracy
    arena_score = compute_arena_score(avg_cost_per_1000, avg_accuracy)

    # Print results
    print("=" * 80)
    print(f"Router: {router_name}")
    print("=" * 80)
    print(f"Total Queries: {num_queries}")
    print(f"Queries with Accuracy: {len(accuracies)}")
    print(f"Queries with Valid Cost: {valid_cost_count}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Total Cost: ${total_cost:.6f}")
    if num_queries > 0:
        print(f"Average Cost per Query: ${total_cost / num_queries:.6f}")
    else:
        print("Average Cost per Query: $0.00")
    print(f"Average Cost per 1K Queries: ${avg_cost_per_1000:.4f}")
    print(f"Arena Score: {arena_score:.4f}")
    print(
        "PLEASE NOTE: The sub_10 dataset is a subset of the full dataset and is used for testing purposes. It is generally easier than the full dataset."
    )
    print("=" * 80)

    return {
        "router_name": router_name,
        "num_queries": num_queries,
        "accuracy": avg_accuracy,
        "total_cost": total_cost,
        "avg_cost_per_query": total_cost / num_queries if num_queries > 0 else 0.0,
        "avg_cost_per_1000": avg_cost_per_1000,
        "arena_score": arena_score,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute arena score for a router based on its prediction file"
    )
    parser.add_argument(
        "router_name",
        type=str,
        help="Name of the router (corresponds to ./router_inference/predictions/<router_name>.json)",
    )

    args = parser.parse_args()

    compute_scores(args.router_name)


if __name__ == "__main__":
    main()
