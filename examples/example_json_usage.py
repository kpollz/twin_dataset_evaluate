#!/usr/bin/env python3
"""
Ví dụ sử dụng pipeline với JSON input
"""

import os
import sys
import json
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pipeline import run_evaluation_pipeline_from_json
from src.data_loader import TwinDatasetLoader, create_balanced_dataset_structure


def create_sample_json_files():
    """
    Tạo file JSON mẫu để test
    """
    print("Creating sample JSON files...")
    
    # Tạo id_images.json mẫu
    id_images_data = {
        "id_1": [
            "/path/to/images/id_1/img1.jpg",
            "/path/to/images/id_1/img2.jpg",
            "/path/to/images/id_1/img3.jpg",
            "/path/to/images/id_1/img4.jpg",
            "/path/to/images/id_1/img5.jpg"
        ],
        "id_2": [
            "/path/to/images/id_2/img1.jpg",
            "/path/to/images/id_2/img2.jpg",
            "/path/to/images/id_2/img3.jpg"
        ],
        "id_3": [
            "/path/to/images/id_3/img1.jpg",
            "/path/to/images/id_3/img2.jpg",
            "/path/to/images/id_3/img3.jpg",
            "/path/to/images/id_3/img4.jpg",
            "/path/to/images/id_3/img5.jpg",
            "/path/to/images/id_3/img6.jpg",
            "/path/to/images/id_3/img7.jpg"
        ],
        "id_4": [
            "/path/to/images/id_4/img1.jpg",
            "/path/to/images/id_4/img2.jpg"
        ]
    }
    
    # Tạo pairs_twin.json mẫu
    pairs_twin_data = [
        ["id_1", "id_2"],
        ["id_3", "id_4"]
    ]
    
    # Lưu files
    with open("sample_id_images.json", "w") as f:
        json.dump(id_images_data, f, indent=2)
    
    with open("sample_pairs_twin.json", "w") as f:
        json.dump(pairs_twin_data, f, indent=2)
    
    print("Sample files created: sample_id_images.json, sample_pairs_twin.json")


def example_json_usage():
    """
    Ví dụ sử dụng với JSON input
    """
    print("=== Example: JSON Input Usage ===")
    
    # Tạo file JSON mẫu
    create_sample_json_files()
    
    # Chạy evaluation với JSON input
    results = run_evaluation_pipeline_from_json(
        id_images_path="sample_id_images.json",
        pairs_twin_path="sample_pairs_twin.json",
        output_dir="json_results",
        model_name="ir_50",
        device_ids=[0],
        batch_size=32,
        balance_strategy="min",
        handle_unequal=True
    )
    
    if results:
        print("Evaluation completed successfully!")
        print(f"Results saved to: json_results/")
    else:
        print("Evaluation failed!")


def example_different_balance_strategies():
    """
    Ví dụ so sánh các chiến lược cân bằng khác nhau
    """
    print("\n=== Example: Different Balance Strategies ===")
    
    strategies = ["min", "max", "random", "all"]
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        try:
            results = run_evaluation_pipeline_from_json(
                id_images_path="sample_id_images.json",
                pairs_twin_path="sample_pairs_twin.json",
                output_dir=f"results_{strategy}",
                balance_strategy=strategy,
                handle_unequal=True
            )
            
            if results:
                overall_metrics = results['overall_metrics']
                print(f"  Overall Score: {overall_metrics['overall_score']:.4f}")
                print(f"  Consistency: {overall_metrics['overall_consistency_mean']:.4f}")
                print(f"  Fidelity: {overall_metrics['overall_fidelity_mean']:.4f}")
            else:
                print(f"  Failed")
                
        except Exception as e:
            print(f"  Error: {e}")


def example_data_loader_usage():
    """
    Ví dụ sử dụng data loader trực tiếp
    """
    print("\n=== Example: Data Loader Usage ===")
    
    # Khởi tạo loader
    loader = TwinDatasetLoader("sample_id_images.json", "sample_pairs_twin.json")
    
    # Load data
    id_images_data, pairs_twin_data = loader.load_data()
    
    # Validate data
    if loader.validate_data():
        print("Data validation passed!")
    else:
        print("Data validation failed!")
        return
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"Dataset Statistics:")
    print(f"  Total IDs: {stats['total_ids']}")
    print(f"  Total Pairs: {stats['total_pairs']}")
    print(f"  Min images per ID: {stats['min_images_per_id']}")
    print(f"  Max images per ID: {stats['max_images_per_id']}")
    print(f"  Avg images per ID: {stats['avg_images_per_id']:.2f}")
    
    # Get twin pairs
    twin_pairs = loader.get_twin_pairs_with_images()
    print(f"Found {len(twin_pairs)} valid twin pairs")
    
    for i, (id1, images1, id2, images2) in enumerate(twin_pairs):
        print(f"  Pair {i+1}: {id1} ({len(images1)} images) vs {id2} ({len(images2)} images)")


def example_handle_unequal_counts():
    """
    Ví dụ xử lý số lượng ảnh không đồng đều
    """
    print("\n=== Example: Handle Unequal Counts ===")
    
    from src.data_loader import handle_unequal_image_counts
    
    # Giả sử có 2 list ảnh với số lượng khác nhau
    images1 = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]  # 5 ảnh
    images2 = ["img1.jpg", "img2.jpg", "img3.jpg"]  # 3 ảnh
    
    strategies = ["min", "max", "random", "all"]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        balanced1, balanced2 = handle_unequal_image_counts(images1, images2, strategy)
        print(f"  Original: {len(images1)} vs {len(images2)}")
        print(f"  Balanced: {len(balanced1)} vs {len(balanced2)}")
        
        if strategy == "min":
            print(f"  Result: Using minimum count ({min(len(images1), len(images2))})")
        elif strategy == "max":
            print(f"  Result: Using maximum count ({max(len(images1), len(images2))})")
        elif strategy == "random":
            print(f"  Result: Random sampling to equal counts")
        elif strategy == "all":
            print(f"  Result: Keeping all images (no balancing)")


def example_create_balanced_dataset():
    """
    Ví dụ tạo dataset structure với cân bằng
    """
    print("\n=== Example: Create Balanced Dataset ===")
    
    strategies = ["min", "max", "random", "all"]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        
        try:
            dataset_structure = create_balanced_dataset_structure(
                "sample_id_images.json",
                "sample_pairs_twin.json",
                strategy
            )
            
            print(f"  Created {len(dataset_structure)} pairs")
            
            for pair_key, pair_data in dataset_structure.items():
                real_count = len(pair_data['real_images'])
                twin_count = len(pair_data['twin_images'])
                original1 = pair_data.get('original_count1', real_count)
                original2 = pair_data.get('original_count2', twin_count)
                
                print(f"    {pair_key}:")
                print(f"      Original: {original1} vs {original2}")
                print(f"      Balanced: {real_count} vs {twin_count}")
                
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    print("Twin Dataset Evaluation Pipeline - JSON Input Examples")
    print("=" * 60)
    
    # Run examples
    example_json_usage()
    example_different_balance_strategies()
    example_data_loader_usage()
    example_handle_unequal_counts()
    example_create_balanced_dataset()
    
    print("\n" + "=" * 60)
    print("JSON examples completed!")
    
    # Clean up sample files
    for file in ["sample_id_images.json", "sample_pairs_twin.json"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up {file}") 