#!/usr/bin/env python3
"""
Integrate multiple batch inference results by averaging attention scores
across different inference runs for common case IDs.
"""

import json
from pathlib import Path


class InferenceResult:
    """Container for a single inference result with its metadata."""

    def __init__(self, name_mapping_path, metadata_path, inference_path):
        """
        Load inference result and associated metadata.

        Args:
            name_mapping_path: Path to activity_name_mapping.json
            metadata_path: Path to metadata.json
            inference_path: Path to inference result JSON (attention_scores.json)
        """
        # Load activity name mapping (normalized -> original)
        with open(name_mapping_path, 'r', encoding='utf-8') as f:
            self.name_mapping = json.load(f)

        # Load metadata (contains x_word_dict and y_word_dict)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # Load inference results (case_id -> {activity: score})
        with open(inference_path, 'r', encoding='utf-8') as f:
            self.inference = json.load(f)

        # Store paths for error messages
        self.inference_path = inference_path

        # Validate that inference activities match metadata
        self.validate()

    def validate(self):
        """
        Validate that activities in inference results match metadata's y_word_dict.

        Raises:
            ValueError: If activity lists don't match
        """
        # Collect all unique activities from inference results
        inference_activities = set()
        for case_data in self.inference.values():
            inference_activities.update(case_data.keys())

        # Get activities from metadata y_word_dict
        metadata_activities = set(self.metadata['y_word_dict'].keys())

        # Check if they match
        if inference_activities != metadata_activities:
            missing_in_inference = metadata_activities - inference_activities
            extra_in_inference = inference_activities - metadata_activities

            error_msg = f"Activity mismatch in {self.inference_path}!\n"
            if missing_in_inference:
                error_msg += f"  Missing in inference: {missing_in_inference}\n"
            if extra_in_inference:
                error_msg += f"  Extra in inference: {extra_in_inference}\n"

            raise ValueError(error_msg)

    def get_case_ids(self):
        """Return set of case IDs in this inference result."""
        return set(self.inference.keys())

    def get_scores_for_case(self, case_id):
        """
        Get attention scores for a specific case ID.

        Args:
            case_id: Case ID to retrieve

        Returns:
            Dictionary of {activity: score}, or empty dict if case not found
        """
        return self.inference.get(case_id, {})

    def normalize_to_original(self, normalized_name):
        """
        Convert normalized activity name to original name.

        Args:
            normalized_name: Normalized activity name (lowercase, dashes)

        Returns:
            Original activity name, or normalized name if mapping not found
        """
        return self.name_mapping.get(normalized_name, normalized_name)


def integrate_inference_results(inference_configs, output_dir):
    """
    Integrate multiple batch inference results by averaging attention scores.

    Args:
        inference_configs: List of dicts with keys:
            - 'activity_name_mapping': Path to activity_name_mapping.json
            - 'metadata': Path to metadata.json
            - 'inference_json': Path to attention_scores.json
        output_dir: Directory to save integrated.json
    """
    print(f"Loading {len(inference_configs)} inference results...")

    # Load all inference results
    results = []
    for i, config in enumerate(inference_configs, 1):
        print(f"  [{i}/{len(inference_configs)}] Loading {config['inference_json']}")
        try:
            result = InferenceResult(
                config['activity_name_mapping'],
                config['metadata'],
                config['inference_json']
            )
            results.append(result)
            print(f"      ✓ Loaded {len(result.get_case_ids())} cases")
        except Exception as e:
            print(f"      ✗ Error: {e}")
            raise

    # Find common case IDs across all results
    print("\nFinding common case IDs...")
    common_case_ids = results[0].get_case_ids()
    for i, result in enumerate(results[1:], 2):
        before = len(common_case_ids)
        common_case_ids &= result.get_case_ids()
        after = len(common_case_ids)
        print(f"  After result {i}: {after} common cases (removed {before - after})")

    common_case_ids = sorted(common_case_ids)  # Sort for consistent output
    print(f"\n✓ Found {len(common_case_ids)} common case IDs across all results")

    if len(common_case_ids) == 0:
        raise ValueError("No common case IDs found across all inference results!")

    # Get all possible activities from first result's metadata
    # All results should have the same activities (validated above)
    all_activities = sorted(results[0].metadata['y_word_dict'].keys())
    print(f"  Activities per case: {len(all_activities)}")

    # Integrate scores for each case
    print("\nIntegrating attention scores...")
    integrated = {}

    for case_id in common_case_ids:
        # Collect scores from all results for this case
        activity_scores = {}

        for activity in all_activities:
            # Collect scores for this activity from all results
            scores = []
            for result in results:
                case_scores = result.get_scores_for_case(case_id)
                # Use 0.0 if activity not present in this result
                score = case_scores.get(activity, 0.0)
                scores.append(score)

            # Average across all results
            # Important: Always divide by total number of results,
            # even if some results don't have this activity (treat as 0)
            avg_score = sum(scores) / len(results)

            # Convert normalized name to original name
            original_name = results[0].normalize_to_original(activity)
            activity_scores[original_name] = avg_score

        integrated[case_id] = activity_scores

    # Save integrated result
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "integrated.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(integrated, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Integrated results saved to: {output_file}")
    print(f"  Total cases: {len(integrated)}")
    print(f"  Activities per case: {len(all_activities)}")


if __name__ == "__main__":
    # Define inference configurations
    # Each element contains paths to the three required files
    inference_configs = [
        {
            'activity_name_mapping': './datasets/processed/bpic2017/activity_name_mapping.json',
            'metadata': './datasets/processed/bpic2017/metadata.json',
            'inference_json': './output/bpic2017_run1/attention_scores.json'
        },
        {
            'activity_name_mapping': './datasets/processed/bpic2017/activity_name_mapping.json',
            'metadata': './datasets/processed/bpic2017/metadata.json',
            'inference_json': './output/bpic2017_run2/attention_scores.json'
        },
        # Add more configurations as needed
    ]

    # Output directory for integrated results
    output_dir = './output/integrated'

    # Run integration
    integrate_inference_results(inference_configs, output_dir)
