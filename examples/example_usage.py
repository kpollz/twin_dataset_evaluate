#!/usr/bin/env python3
"""
Ví dụ sử dụng pipeline đánh giá bộ dữ liệu khuôn mặt song sinh
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pipeline import TwinDatasetEvaluationPipeline, create_dataset_structure_from_directories
from src.face_embedding import AdaFaceEmbeddingExtractor
from src.metrics import TwinDatasetEvaluator


def example_basic_usage():
    """
    Ví dụ cơ bản về cách sử dụng pipeline
    """
    print("=== Example: Basic Pipeline Usage ===")
    
    # Giả sử bạn có dữ liệu với structure:
    # data/
    #   real_images/
    #     person_1/
    #       img1.jpg, img2.jpg, ...
    #     person_2/
    #       img1.jpg, img2.jpg, ...
    #   twin_images/
    #     person_1/
    #       img1_twin.jpg, img2_twin.jpg, ...
    #     person_2/
    #       img1_twin.jpg, img2_twin.jpg, ...
    
    real_dir = "data/real_images"
    twin_dir = "data/twin_images"
    output_dir = "results"
    
    # Tạo dataset structure
    dataset_structure = create_dataset_structure_from_directories(real_dir, twin_dir)
    
    if len(dataset_structure) == 0:
        print("No valid dataset found. Please check your directory structure.")
        return
    
    # Initialize pipeline
    pipeline = TwinDatasetEvaluationPipeline(
        model_name="ir_50",
        device_ids=[0, 1],  # Sử dụng GPU 0 và 1
        batch_size=32
    )
    
    # Run evaluation
    results = pipeline.evaluate_dataset(dataset_structure, output_dir)
    
    # Print results
    overall_metrics = results['overall_metrics']
    print(f"Overall Consistency Score: {overall_metrics['overall_consistency_mean']:.4f}")
    print(f"Overall Fidelity Score: {overall_metrics['overall_fidelity_mean']:.4f}")
    print(f"Overall Score: {overall_metrics['overall_score']:.4f}")


def example_optimized_embedding_extraction():
    """
    Ví dụ về cách tối ưu embedding extraction
    """
    print("\n=== Example: Optimized Embedding Extraction ===")
    
    # Initialize embedding extractor với multi-GPU
    extractor = AdaFaceEmbeddingExtractor(
        model_name="ir_50",
        device_ids=[0, 1, 2, 3]  # Sử dụng 4 GPUs
    )
    
    # Giả sử bạn có list ảnh
    # Trong thực tế, bạn sẽ load từ file
    dummy_images = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) 
                   for _ in range(100)]
    
    # Extract embeddings với batch processing
    embeddings = extractor.extract_embeddings_batch(dummy_images, batch_size=64)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Compute similarity matrix tối ưu
    similarity_matrix = extractor.compute_similarity_matrix(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Compute pairwise similarity
    embeddings_2 = np.random.rand(50, embeddings.shape[1])
    pairwise_similarity = extractor.compute_pairwise_similarity(embeddings, embeddings_2)
    print(f"Pairwise similarity shape: {pairwise_similarity.shape}")


def example_metrics_calculation():
    """
    Ví dụ về cách tính các metrics
    """
    print("\n=== Example: Metrics Calculation ===")
    
    evaluator = TwinDatasetEvaluator()
    
    # Giả sử bạn có similarity matrices
    # Trong thực tế, đây sẽ là kết quả từ embedding extraction
    n_images = 10
    twin_similarity = np.random.rand(n_images, n_images)
    real_twin_similarity = np.random.rand(n_images, n_images)
    
    # Đảm bảo matrices là symmetric và có diagonal = 1
    twin_similarity = (twin_similarity + twin_similarity.T) / 2
    np.fill_diagonal(twin_similarity, 1.0)
    
    real_twin_similarity = (real_twin_similarity + real_twin_similarity.T) / 2
    np.fill_diagonal(real_twin_similarity, 1.0)
    
    # Tính metrics
    consistency_metrics = evaluator.compute_consistency_metrics(twin_similarity)
    fidelity_metrics = evaluator.compute_fidelity_metrics(real_twin_similarity)
    
    print("Consistency Metrics:")
    for key, value in consistency_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nFidelity Metrics:")
    for key, value in fidelity_metrics.items():
        print(f"  {key}: {value:.4f}")


def example_batch_processing_optimization():
    """
    Ví dụ về tối ưu batch processing
    """
    print("\n=== Example: Batch Processing Optimization ===")
    
    # Test với các batch sizes khác nhau
    batch_sizes = [16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Initialize pipeline với batch size cụ thể
        pipeline = TwinDatasetEvaluationPipeline(
            model_name="ir_50",
            device_ids=[0],
            batch_size=batch_size
        )
        
        # Giả sử có 1000 ảnh
        dummy_images = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) 
                       for _ in range(1000)]
        
        # Measure time (trong thực tế bạn sẽ dùng time.time())
        embeddings = pipeline.extract_embeddings_optimized(dummy_images)
        print(f"  Extracted {len(embeddings)} embeddings")
        print(f"  Embedding shape: {embeddings.shape}")


def example_multi_gpu_optimization():
    """
    Ví dụ về tối ưu multi-GPU
    """
    print("\n=== Example: Multi-GPU Optimization ===")
    
    import torch
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus > 0:
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Test với số GPU khác nhau
    for num_gpus_to_use in range(1, min(num_gpus + 1, 5)):
        print(f"\nTesting with {num_gpus_to_use} GPU(s)")
        
        device_ids = list(range(num_gpus_to_use))
        
        try:
            pipeline = TwinDatasetEvaluationPipeline(
                model_name="ir_50",
                device_ids=device_ids,
                batch_size=32 * num_gpus_to_use  # Scale batch size với số GPU
            )
            
            # Test với dummy data
            dummy_images = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) 
                           for _ in range(100)]
            
            embeddings = pipeline.extract_embeddings_optimized(dummy_images)
            print(f"  Successfully extracted embeddings with {num_gpus_to_use} GPU(s)")
            
        except Exception as e:
            print(f"  Error with {num_gpus_to_use} GPU(s): {e}")


if __name__ == "__main__":
    print("Twin Dataset Evaluation Pipeline Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_optimized_embedding_extraction()
    example_metrics_calculation()
    example_batch_processing_optimization()
    example_multi_gpu_optimization()
    
    print("\n" + "=" * 50)
    print("Examples completed!") 