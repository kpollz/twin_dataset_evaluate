#!/usr/bin/env python3
"""
Ví dụ minh họa xử lý số lượng ảnh không đồng đều
"""

import os
import sys
import json
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pipeline import run_evaluation_pipeline_from_json
from src.metrics import TwinDatasetEvaluator
from src.data_loader import TwinDatasetLoader


def create_unequal_sample_data():
    """
    Tạo dữ liệu mẫu với số lượng ảnh không đồng đều
    """
    print("Creating sample data with unequal image counts...")
    
    # Tạo id_images.json với số lượng ảnh khác nhau
    id_images_data = {
        "id_1": [f"/path/to/images/id_1/img{i}.jpg" for i in range(1, 6)],      # 5 ảnh
        "id_2": [f"/path/to/images/id_2/img{i}.jpg" for i in range(1, 11)],     # 10 ảnh
        "id_3": [f"/path/to/images/id_3/img{i}.jpg" for i in range(1, 4)],      # 3 ảnh
        "id_4": [f"/path/to/images/id_4/img{i}.jpg" for i in range(1, 16)],     # 15 ảnh
        "id_5": [f"/path/to/images/id_5/img{i}.jpg" for i in range(1, 8)],      # 7 ảnh
        "id_6": [f"/path/to/images/id_6/img{i}.jpg" for i in range(1, 12)]      # 11 ảnh
    }
    
    # Tạo pairs_twin.json với các cặp có số lượng khác nhau
    pairs_twin_data = [
        ["id_1", "id_2"],  # 5 vs 10 ảnh
        ["id_3", "id_4"],  # 3 vs 15 ảnh
        ["id_5", "id_6"]   # 7 vs 11 ảnh
    ]
    
    # Lưu files
    with open("unequal_id_images.json", "w") as f:
        json.dump(id_images_data, f, indent=2)
    
    with open("unequal_pairs_twin.json", "w") as f:
        json.dump(pairs_twin_data, f, indent=2)
    
    print("Sample files created: unequal_id_images.json, unequal_pairs_twin.json")
    
    return id_images_data, pairs_twin_data


def example_unequal_counts_processing():
    """
    Ví dụ xử lý số lượng ảnh không đồng đều
    """
    print("=== Example: Unequal Counts Processing ===")
    
    # Tạo dữ liệu mẫu
    id_images_data, pairs_twin_data = create_unequal_sample_data()
    
    # Chạy evaluation với handle_unequal=True
    results = run_evaluation_pipeline_from_json(
        id_images_path="unequal_id_images.json",
        pairs_twin_path="unequal_pairs_twin.json",
        output_dir="unequal_results",
        model_name="ir_50",
        device_ids=[0],
        batch_size=32,
        balance_strategy="all",  # Không cân bằng
        handle_unequal=True
    )
    
    if results:
        print("Evaluation completed successfully!")
        print(f"Results saved to: unequal_results/")
        
        # In thống kê về ma trận
        for i, metrics in enumerate(results['individual_metrics']):
            if 'matrix_shape' in metrics:
                print(f"Pair {i+1}: Matrix {metrics['matrix_shape']}")
                print(f"  Real images: {metrics['num_real_images']}")
                print(f"  Twin images: {metrics['num_twin_images']}")
    else:
        print("Evaluation failed!")


def example_metrics_comparison():
    """
    So sánh metrics giữa các phương pháp khác nhau
    """
    print("\n=== Example: Metrics Comparison ===")
    
    # Tạo dữ liệu mẫu
    create_unequal_sample_data()
    
    # Test với các balance strategies khác nhau
    strategies = ["min", "max", "random", "all"]
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        try:
            results = run_evaluation_pipeline_from_json(
                id_images_path="unequal_id_images.json",
                pairs_twin_path="unequal_pairs_twin.json",
                output_dir=f"results_{strategy}",
                balance_strategy=strategy,
                handle_unequal=True
            )
            
            if results:
                overall_metrics = results['overall_metrics']
                print(f"  Overall Score: {overall_metrics['overall_score']:.4f}")
                print(f"  Consistency: {overall_metrics['overall_consistency_mean']:.4f}")
                print(f"  Fidelity: {overall_metrics['overall_fidelity_mean']:.4f}")
                
                # In thông tin về ma trận
                for i, metrics in enumerate(results['individual_metrics']):
                    if 'matrix_shape' in metrics:
                        print(f"    Pair {i+1}: {metrics['matrix_shape']}")
            else:
                print(f"  Failed")
                
        except Exception as e:
            print(f"  Error: {e}")


def example_matrix_visualization():
    """
    Ví dụ visualization cho ma trận không vuông
    """
    print("\n=== Example: Matrix Visualization ===")
    
    from src.metrics import TwinDatasetEvaluator
    
    evaluator = TwinDatasetEvaluator()
    
    # Tạo ma trận similarity mẫu với kích thước khác nhau
    n_real, n_twin = 5, 10
    similarity_matrix = np.random.rand(n_real, n_twin)
    
    # Normalize để có giá trị từ 0-1
    similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
    
    print(f"Creating heatmap for matrix: {n_real}×{n_twin}")
    
    # Tạo heatmap
    evaluator.create_unequal_similarity_heatmap(
        similarity_matrix,
        f"Sample Unequal Similarity Matrix ({n_real}×{n_twin})",
        "unequal_heatmap.png"
    )


def example_fidelity_metrics_explanation():
    """
    Giải thích các fidelity metrics cho trường hợp không đồng đều
    """
    print("\n=== Example: Fidelity Metrics Explanation ===")
    
    # Tạo embeddings mẫu
    n_real, n_twin = 5, 10
    embedding_dim = 512
    
    real_embeddings = np.random.rand(n_real, embedding_dim)
    twin_embeddings = np.random.rand(n_twin, embedding_dim)
    
    # Normalize embeddings
    real_embeddings = real_embeddings / np.linalg.norm(real_embeddings, axis=1, keepdims=True)
    twin_embeddings = twin_embeddings / np.linalg.norm(twin_embeddings, axis=1, keepdims=True)
    
    # Tính similarity matrix
    similarity_matrix = np.dot(real_embeddings, twin_embeddings.T)
    
    print(f"Similarity Matrix Shape: {similarity_matrix.shape}")
    print(f"Real Images: {n_real}")
    print(f"Twin Images: {n_twin}")
    
    # Tính các metrics
    from src.metrics import TwinDatasetEvaluator
    evaluator = TwinDatasetEvaluator()
    
    metrics = evaluator.compute_fidelity_metrics_unequal_counts(real_embeddings, twin_embeddings)
    
    print("\nFidelity Metrics:")
    print(f"  1. Mean All: {metrics['fidelity_mean_all']:.4f}")
    print(f"  2. Mean Max Per Real: {metrics['fidelity_mean_max_per_real']:.4f}")
    print(f"  3. Mean Max Per Twin: {metrics['fidelity_mean_max_per_twin']:.4f}")
    print(f"  4. Mean Top-K Per Real: {metrics['fidelity_mean_top_k_per_real']:.4f}")
    print(f"  5. Mean Top-K Per Twin: {metrics['fidelity_mean_top_k_per_twin']:.4f}")
    print(f"  6. Fidelity Score: {metrics['fidelity_score']:.4f}")
    
    print(f"\nMatrix Info:")
    print(f"  Shape: {metrics['matrix_shape']}")
    print(f"  Real Images: {metrics['num_real_images']}")
    print(f"  Twin Images: {metrics['num_twin_images']}")


def example_why_no_balancing_needed():
    """
    Giải thích tại sao không cần cân bằng
    """
    print("\n=== Example: Why No Balancing Needed ===")
    
    print("1. CONSISTENCY METRICS:")
    print("   - Tính trong nội bộ mỗi ID (ma trận vuông)")
    print("   - Không bị ảnh hưởng bởi số lượng ảnh khác nhau")
    print("   - Ví dụ: ID1 có 5 ảnh → ma trận 5×5")
    print("   - Ví dụ: ID2 có 10 ảnh → ma trận 10×10")
    
    print("\n2. FIDELITY METRICS:")
    print("   - Tính từ ma trận N1×N2 (không vuông)")
    print("   - Có thể tính các metrics khác nhau:")
    print("     * Mean similarity toàn bộ")
    print("     * Max similarity cho mỗi ảnh gốc")
    print("     * Max similarity cho mỗi ảnh song sinh")
    print("     * Top-k similarity")
    
    print("\n3. ADVANTAGES OF NO BALANCING:")
    print("   - Giữ nguyên tất cả thông tin")
    print("   - Không mất mát dữ liệu")
    print("   - Phản ánh đúng thực tế")
    print("   - Metrics chính xác hơn")
    
    print("\n4. MATRIX SHAPES:")
    print("   - Real vs Real: N1×N1 (vuông)")
    print("   - Twin vs Twin: N2×N2 (vuông)")
    print("   - Real vs Twin: N1×N2 (không vuông)")
    
    print("\n5. HEATMAP VISUALIZATION:")
    print("   - Ma trận vuông: heatmap vuông")
    print("   - Ma trận không vuông: heatmap hình chữ nhật")
    print("   - Hiển thị đầy đủ thông tin")


if __name__ == "__main__":
    print("Twin Dataset Evaluation Pipeline - Unequal Counts Examples")
    print("=" * 60)
    
    # Run examples
    example_unequal_counts_processing()
    example_metrics_comparison()
    example_matrix_visualization()
    example_fidelity_metrics_explanation()
    example_why_no_balancing_needed()
    
    print("\n" + "=" * 60)
    print("Unequal counts examples completed!")
    
    # Clean up sample files
    for file in ["unequal_id_images.json", "unequal_pairs_twin.json"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up {file}") 