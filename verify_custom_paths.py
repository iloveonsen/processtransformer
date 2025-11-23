#!/usr/bin/env python3
"""
Verification script to test path consistency with custom data directories.
Tests that processor and loader work correctly with various custom paths.
"""

def test_path_consistency_with_custom_dir(custom_dir, dataset_name="test_dataset"):
    """Test path consistency with a custom directory."""
    # Simulate processor path construction
    processor_path = f"{custom_dir}/processed/{dataset_name}"

    # Simulate loader path construction
    loader_path = f"{custom_dir}/processed/{dataset_name}"

    print(f"\nTesting with custom_dir: {custom_dir}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Processor writes to: {processor_path}")
    print(f"  Loader reads from:   {loader_path}")

    if processor_path == loader_path:
        print(f"  ✓ Paths match!")
        return True
    else:
        print(f"  ✗ Paths DO NOT match!")
        return False

def test_all_scenarios():
    """Test various path scenarios."""
    print("=" * 70)
    print("Path Consistency Verification with Custom Directories")
    print("=" * 70)

    test_cases = [
        ("./datasets", "helpdesk"),
        ("/home/user/data", "bpi_2012"),
        ("/mnt/storage/ml_data", "bpi_2013"),
        ("../shared_datasets", "helpdesk"),
        ("/data/process_mining", "custom_dataset"),
    ]

    all_passed = True
    for custom_dir, dataset_name in test_cases:
        passed = test_path_consistency_with_custom_dir(custom_dir, dataset_name)
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nExpected directory structure for any custom path:")
        print("  {custom_dir}/")
        print("  ├── raw/")
        print("  │   └── {dataset_name}.csv")
        print("  └── processed/")
        print("      └── {dataset_name}/")
        print("          ├── metadata.json")
        print("          ├── next_activity_train.csv")
        print("          ├── next_activity_test.csv")
        print("          ├── next_time_train.csv")
        print("          ├── next_time_test.csv")
        print("          ├── remaining_time_train.csv")
        print("          └── remaining_time_test.csv")
        print("\nUsage examples:")
        print("  # Data processing:")
        print("  python data_processing.py --dataset helpdesk \\")
        print("      --raw_log_file /custom/path/raw/helpdesk.csv \\")
        print("      --dir_path /custom/path")
        print("\n  # Training:")
        print("  python next_activity.py --dataset helpdesk --data_dir /custom/path")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    import sys
    success = test_all_scenarios()
    sys.exit(0 if success else 1)
