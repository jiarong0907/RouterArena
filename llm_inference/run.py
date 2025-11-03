# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Run LLM inference for router predictions.

This script processes router prediction files and makes LLM API calls
for each prediction, using cached results when available.
"""

import argparse
import json
import os
import sys
import logging
import datetime
import time
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../"))
os.chdir(base_dir)

# Load environment variables from .env file
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from model_inference import ModelInference  # noqa: E402
from universal_model_names import ModelNameManager  # noqa: E402
from pipeline import load_jsonl_file  # noqa: E402

logger = logging.getLogger(__name__)


def load_predictions_file(router_name: str) -> List[Dict[str, Any]]:
    """
    Load router predictions from JSON file.

    Args:
        router_name: Name of the router

    Returns:
        List of prediction dictionaries
    """
    prediction_path = f"./router_inference/predictions/{router_name}.json"

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}\n"
            f"Please create the prediction file first. See README.md for format."
        )

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    logger.info(f"Loaded {len(predictions)} predictions from {prediction_path}")
    return predictions


def find_cached_result(
    global_index: str, model_name: str, cached_results_dir: str = "./cached_results"
) -> Optional[Dict[str, Any]]:
    """
    Check if a result already exists in cached results for the given global_index and model.

    Args:
        global_index: Global index of the query
        model_name: Universal model name
        cached_results_dir: Directory containing cached results

    Returns:
        Cached result dictionary if found and successful, None otherwise
    """
    cached_file = os.path.join(cached_results_dir, f"{model_name}.jsonl")

    if not os.path.exists(cached_file):
        return None

    # Load cached results
    cached_results = load_jsonl_file(cached_file)

    # Find matching entry by global_index
    for result in cached_results:
        if result.get("global_index") == global_index:
            # Only return if successful
            if result.get("success", False):
                return result

    return None


def save_to_cached_results(
    result: Dict[str, Any],
    model_name: str,
    cached_results_dir: str = "./cached_results",
) -> None:
    """
    Save or append a result to the cached results file.

    Args:
        result: Result dictionary to save
        model_name: Universal model name
        cached_results_dir: Directory to save cached results
    """
    os.makedirs(cached_results_dir, exist_ok=True)
    cached_file = os.path.join(cached_results_dir, f"{model_name}.jsonl")

    # Load existing results to avoid duplicates
    existing_results = load_jsonl_file(cached_file)

    # Only append if this global_index doesn't exist, or update if it does
    updated = False
    for i, existing in enumerate(existing_results):
        if existing.get("global_index") == result.get("global_index"):
            existing_results[i] = result
            updated = True
            break

    if not updated:
        existing_results.append(result)

    # Write all results back
    with open(cached_file, "w", encoding="utf-8") as f:
        for res in existing_results:
            json.dump(res, f, ensure_ascii=False)
            f.write("\n")


def save_predictions_file(
    predictions: List[Dict[str, Any]], router_name: str, create_backup: bool = False
) -> None:
    """
    Save predictions back to file.

    Args:
        predictions: List of prediction dictionaries
        router_name: Name of the router
        create_backup: Whether to create a backup before saving (only needed once)
    """
    prediction_path = f"./router_inference/predictions/{router_name}.json"

    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.debug(f"Saved predictions to {prediction_path}")


def process_router_predictions(router_name: str) -> None:
    """
    Process router predictions by making LLM calls for each entry.

    Args:
        router_name: Name of the router
    """
    logger.info(f"Starting LLM inference for router: {router_name}")

    # Load predictions
    predictions = load_predictions_file(router_name)

    # Create backup of original predictions file
    save_predictions_file(predictions, router_name, create_backup=True)

    # Initialize model inference
    model_inferencer = ModelInference()

    # Statistics
    total = len(predictions)
    skipped_count = 0
    cached_count = 0
    successful_count = 0
    failed_count = 0

    start_time = datetime.datetime.now()

    # Process each prediction
    for i, prediction in enumerate(predictions):
        global_index = prediction.get("global index") or prediction.get("global_index")
        prompt = prediction.get("prompt", "")
        model_name = prediction.get("prediction", "")
        existing_result = prediction.get("generated_result")

        if not global_index:
            logger.warning(f"Skipping entry {i + 1}: missing global_index")
            continue

        if not prompt:
            logger.warning(
                f"Skipping entry {i + 1} (global_index: {global_index}): missing prompt"
            )
            continue

        if not model_name:
            logger.warning(
                f"Skipping entry {i + 1} (global_index: {global_index}): missing prediction/model name"
            )
            continue

        # Skip if already has a successful result
        if existing_result and isinstance(existing_result, dict):
            if existing_result.get("success", False):
                logger.debug(
                    f"⏭️  Skipping {global_index}: already has successful generated_result"
                )
                skipped_count += 1
                continue

        # Convert to universal model name for cache lookup and llm_selected field
        # Note: We'll use original model_name for inference (like pipeline.py does)
        try:
            universal_model_name = ModelNameManager.get_universal_name(model_name)
        except Exception as e:
            logger.error(
                f"Error converting model name '{model_name}' to universal name: {e}\n"
                f"Make sure the model is in universal_model_names.py"
            )
            failed_count += 1
            continue

        logger.info(
            f"Processing {i + 1}/{total} | Global Index: {global_index} | "
            f"Model: {universal_model_name}"
        )

        # Check for cached result
        cached_result = find_cached_result(global_index, universal_model_name)

        if cached_result:
            logger.info(
                f"✓ Using cached result for {global_index} (model: {universal_model_name})"
            )
            result_entry = cached_result
            cached_count += 1
        else:
            # Make API call
            logger.info(
                f"🔄 Making API call for {global_index} (model: {universal_model_name})"
            )
            inference_start = datetime.datetime.now()

            try:
                # Use original model_name for inference (like pipeline.py does)
                # The ModelInference class expects model names that match its provider mapping
                # If model_name fails, try universal_model_name as fallback
                try:
                    inference_result = model_inferencer.infer(model_name, prompt)
                except ValueError as ve:
                    # If original name not recognized, try universal name
                    if "not found in model_to_provider" in str(ve):
                        logger.warning(
                            f"Original model name '{model_name}' not recognized, "
                            f"trying universal name '{universal_model_name}'"
                        )
                        inference_result = model_inferencer.infer(
                            universal_model_name, prompt
                        )
                    else:
                        raise
                inference_duration = (
                    datetime.datetime.now() - inference_start
                ).total_seconds()

                # Create result entry matching cached_results golden standard structure
                # Use universal_model_name for llm_selected to match cache structure
                result_entry = {
                    "global_index": global_index,
                    "question": prompt,
                    "llm_selected": universal_model_name,  # Use universal for consistency
                    "generated_answer": inference_result.get("response", ""),
                    "token_usage": inference_result.get(
                        "token_usage",
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    ),
                    "success": inference_result.get("success", False),
                    "provider": inference_result.get("provider", "unknown"),
                    "error": inference_result.get("error", None)
                    if not inference_result.get("success", False)
                    else None,
                    "batch_size": None,  # Match pipeline.py structure
                    "batch_number": None,
                    "batch_index": None,
                    "processing_mode": "individual",
                    "evaluation_result": None,  # Will be filled by evaluation step
                }

                if result_entry["success"]:
                    token_usage = result_entry["token_usage"]
                    logger.info(
                        f"✅ {global_index} | {result_entry['provider']} | "
                        f"Duration: {inference_duration:.2f}s | "
                        f"Tokens: {token_usage.get('input_tokens', 0)}/"
                        f"{token_usage.get('output_tokens', 0)}/"
                        f"{token_usage.get('total_tokens', 0)}"
                    )
                    successful_count += 1
                else:
                    error_msg = result_entry.get("error", "Unknown error")
                    logger.error(
                        f"❌ {global_index} | {result_entry['provider']} | "
                        f"Duration: {inference_duration:.2f}s | Error: {error_msg}"
                    )
                    failed_count += 1

                # Save to cached results
                save_to_cached_results(result_entry, universal_model_name)

            except Exception as e:
                inference_duration = (
                    datetime.datetime.now() - inference_start
                ).total_seconds()
                logger.error(
                    f"💥 EXCEPTION | Global Index: {global_index} | "
                    f"Duration: {inference_duration:.2f}s | Error: {str(e)}"
                )
                failed_count += 1

                # Create failed result entry matching cached_results golden standard structure
                result_entry = {
                    "global_index": global_index,
                    "question": prompt,
                    "llm_selected": universal_model_name,  # Use universal for consistency
                    "generated_answer": "",
                    "token_usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "success": False,
                    "provider": "unknown",
                    "error": str(e),
                    "batch_size": None,  # Match pipeline.py structure
                    "batch_number": None,
                    "batch_index": None,
                    "processing_mode": "individual",
                    "evaluation_result": None,
                }

                # Save failed result to cache
                save_to_cached_results(result_entry, universal_model_name)

        # Update prediction entry with generated_result
        # Store the generated_answer and success status
        prediction["generated_result"] = {
            "generated_answer": result_entry.get("generated_answer", ""),
            "success": result_entry.get("success", False),
            "token_usage": result_entry.get("token_usage", {}),
            "provider": result_entry.get("provider", "unknown"),
            "error": result_entry.get("error", None),
        }

        # Save predictions incrementally after each query (no backup needed)
        save_predictions_file(predictions, router_name, create_backup=False)

        # Progress update every 10 items
        if (i + 1) % 10 == 0:
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds() / 60
            logger.info(
                f"Progress: {i + 1}/{total} processed | "
                f"Skipped: {skipped_count} | Cached: {cached_count} | "
                f"Success: {successful_count} | Failed: {failed_count} | "
                f"Elapsed: {elapsed_time:.1f}min"
            )

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    # Final summary
    end_time = datetime.datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60

    logger.info("=" * 60)
    logger.info("Processing completed!")
    logger.info(
        f"Total: {total} | Skipped: {skipped_count} | Cached: {cached_count} | "
        f"New Success: {successful_count} | New Failed: {failed_count}"
    )
    logger.info(f"Total duration: {total_duration:.1f} minutes")
    logger.info(
        f"Predictions saved to: ./router_inference/predictions/{router_name}.json"
    )
    logger.info("=" * 60)


def main():
    """Main function to handle command line arguments and run inference."""
    parser = argparse.ArgumentParser(
        description="Run LLM inference for router predictions"
    )
    parser.add_argument(
        "router_name",
        type=str,
        help="Name of the router (corresponds to ./router_inference/predictions/<router_name>.json)",
    )
    parser.add_argument(
        "--cached-results-dir",
        type=str,
        default="./cached_results",
        help="Directory containing cached results (default: ./cached_results)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up environment (change to project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "../"))
    os.chdir(base_dir)

    # Run inference
    try:
        process_router_predictions(args.router_name)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Partial results have been saved.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
