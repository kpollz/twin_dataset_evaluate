import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import cv2
from pathlib import Path

from .face_embedding import AdaFaceEmbeddingExtractor, load_images_from_directory
from .metrics import TwinDatasetEvaluator
from .data_loader import TwinDatasetLoader, create_balanced_dataset_structure


class TwinDatasetEvaluationPipeline:
    """
    Pipeline tối ưu để đánh giá bộ dữ liệu khuôn mặt song sinh
    """
    
    def __init__(self, 
                 model_name: str = "ir_50",
                 device_ids: List[int] = None,
                 batch_size: int = 32):
        """
        Args:
            model_name: Tên model AdaFace
            device_ids: List GPU IDs
            batch_size: Kích thước batch cho processing
        """
        self.batch_size = batch_size
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        
        # Initialize components
        self.embedding_extractor = AdaFaceEmbeddingExtractor(
            model_name=model_name,
            device_ids=self.device_ids
        )
        self.evaluator = TwinDatasetEvaluator()
        
        print(f"Pipeline initialized with {len(self.device_ids)} GPUs")
        print(f"Batch size: {self.batch_size}")
    
    def load_and_preprocess_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Load và preprocess tất cả ảnh
        
        Args:
            image_paths: List các path đến ảnh
            
        Returns:
            images: List các ảnh đã preprocess
        """
        images = []
        print(f"Loading {len(image_paths)} images...")
        
        for path in tqdm(image_paths, desc="Loading images"):
            try:
                image = cv2.imread(path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                else:
                    print(f"Warning: Could not load image {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        print(f"Successfully loaded {len(images)} images")
        return images
    
    def extract_embeddings_optimized(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings với tối ưu multi-GPU và batch processing
        
        Args:
            images: List các ảnh
            
        Returns:
            embeddings: Array shape (N, embedding_dim)
        """
        print(f"Extracting embeddings for {len(images)} images...")
        
        # Sử dụng batch processing
        embeddings = self.embedding_extractor.extract_embeddings_batch(
            images, 
            batch_size=self.batch_size
        )
        
        print(f"Extracted embeddings shape: {embeddings.shape}")
        return embeddings
    
    def compute_similarity_matrices(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Tính similarity matrix tối ưu bằng matrix multiplication
        
        Args:
            embeddings: Array shape (N, embedding_dim)
            
        Returns:
            similarity_matrix: Array shape (N, N)
        """
        print("Computing similarity matrix...")
        
        # Sử dụng matrix multiplication tối ưu
        similarity_matrix = self.embedding_extractor.compute_similarity_matrix(embeddings)
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        return similarity_matrix
    
    def evaluate_single_pair(self, 
                           real_images: List[np.ndarray],
                           twin_images: List[np.ndarray],
                           pair_id: str = "pair_1",
                           handle_unequal: bool = True) -> Dict[str, float]:
        """
        Đánh giá cho một cặp song sinh
        
        Args:
            real_images: Ảnh của ID 1 (coi như real)
            twin_images: Ảnh của ID 2 (coi như twin)
            pair_id: ID của cặp
            handle_unequal: Có xử lý số lượng ảnh không đồng đều không
            
        Returns:
            metrics: Dict chứa các metrics
        """
        print(f"\nEvaluating {pair_id}...")
        print(f"  Real images: {len(real_images)}")
        print(f"  Twin images: {len(twin_images)}")
        
        # Extract embeddings
        real_embeddings = self.extract_embeddings_optimized(real_images)
        twin_embeddings = self.extract_embeddings_optimized(twin_images)
        
        # Luôn xử lý trường hợp không đồng đều nếu handle_unequal=True
        if handle_unequal:
            print(f"  Processing unequal counts: {real_embeddings.shape[0]} vs {twin_embeddings.shape[0]}")
            
            # Tính metrics cho trường hợp không đồng đều
            consistency_metrics = self.evaluator.compute_consistency_metrics(
                np.dot(twin_embeddings, twin_embeddings.T)
            )
            fidelity_metrics = self.evaluator.compute_fidelity_metrics_unequal_counts(
                real_embeddings, twin_embeddings
            )
            diversity_metrics = self.evaluator.compute_diversity_metrics(twin_embeddings)
            baseline_metrics = self.evaluator.compute_baseline_metrics_unequal_counts(
                real_embeddings, twin_embeddings
            )
        else:
            # Xử lý trường hợp số lượng ảnh bằng nhau
            # Compute similarity matrices
            real_similarity = self.compute_similarity_matrices(real_embeddings)
            twin_similarity = self.compute_similarity_matrices(twin_embeddings)
            
            # Compute pairwise similarity between real and twin
            real_twin_similarity = self.embedding_extractor.compute_pairwise_similarity(
                real_embeddings, twin_embeddings
            )
            
            # Calculate metrics
            consistency_metrics = self.evaluator.compute_consistency_metrics(twin_similarity)
            fidelity_metrics = self.evaluator.compute_fidelity_metrics(real_twin_similarity)
            diversity_metrics = self.evaluator.compute_diversity_metrics(twin_embeddings)
            baseline_metrics = self.evaluator.compute_baseline_metrics(
                real_similarity, twin_similarity
            )
        
        # Combine all metrics
        metrics = {
            'pair_id': pair_id,
            **consistency_metrics,
            **fidelity_metrics,
            **diversity_metrics,
            **baseline_metrics
        }
        
        print(f"Results for {pair_id}:")
        print(f"  Consistency Score: {metrics['consistency_score']:.4f}")
        print(f"  Fidelity Score: {metrics['fidelity_score']:.4f}")
        print(f"  Diversity Score: {metrics['diversity_score']:.4f}")
        
        if 'matrix_shape' in metrics:
            print(f"  Matrix Shape: {metrics['matrix_shape']}")
        
        return metrics
    
    def evaluate_dataset_from_json(self, 
                                 id_images_path: str,
                                 pairs_twin_path: str,
                                 output_dir: str = "evaluation_results",
                                 balance_strategy: str = "min",
                                 handle_unequal: bool = True) -> Dict[str, any]:
        """
        Đánh giá dataset từ JSON files
        
        Args:
            id_images_path: Path đến id_images.json
            pairs_twin_path: Path đến pairs_twin.json
            output_dir: Directory để lưu kết quả
            balance_strategy: Chiến lược cân bằng ("min", "max", "random", "all")
            handle_unequal: Có xử lý số lượng ảnh không đồng đều không
            
        Returns:
            results: Dict chứa tất cả kết quả
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load và validate data
        loader = TwinDatasetLoader(id_images_path, pairs_twin_path)
        loader.load_data()
        
        if not loader.validate_data():
            raise ValueError("Invalid data format")
        
        # Get statistics
        stats = loader.get_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total IDs: {stats['total_ids']}")
        print(f"  Total Pairs: {stats['total_pairs']}")
        print(f"  Min images per ID: {stats['min_images_per_id']}")
        print(f"  Max images per ID: {stats['max_images_per_id']}")
        print(f"  Avg images per ID: {stats['avg_images_per_id']:.2f}")
        print(f"  Std images per ID: {stats['std_images_per_id']:.2f}")
        
        # Create dataset structure
        if balance_strategy != "all":
            dataset_structure = create_balanced_dataset_structure(
                id_images_path, pairs_twin_path, balance_strategy
            )
        else:
            # Sử dụng tất cả ảnh, không cân bằng
            dataset_structure = create_dataset_structure_from_json(
                id_images_path, pairs_twin_path
            )
        
        all_metrics = []
        all_similarity_matrices = {}
        
        print(f"\nEvaluating {len(dataset_structure)} twin pairs...")
        
        for pair_key, pair_data in dataset_structure.items():
            print(f"\nProcessing {pair_key}...")
            
            # Load images
            real_images = self.load_and_preprocess_images(pair_data['real_images'])
            twin_images = self.load_and_preprocess_images(pair_data['twin_images'])
            
            if len(real_images) == 0 or len(twin_images) == 0:
                print(f"Warning: No images loaded for {pair_key}")
                continue
            
            # Evaluate pair
            metrics = self.evaluate_single_pair(
                real_images, twin_images, pair_key, handle_unequal
            )
            
            # Add metadata
            metrics.update({
                'id1': pair_data.get('id1', ''),
                'id2': pair_data.get('id2', ''),
                'original_count1': pair_data.get('original_count1', len(real_images)),
                'original_count2': pair_data.get('original_count2', len(twin_images)),
                'balanced_count': pair_data.get('balanced_count', len(real_images))
            })
            
            all_metrics.append(metrics)
            
            # Save similarity matrices for visualization
            real_embeddings = self.extract_embeddings_optimized(real_images)
            twin_embeddings = self.extract_embeddings_optimized(twin_images)
            
            all_similarity_matrices[pair_key] = {
                'real_similarity': self.compute_similarity_matrices(real_embeddings),
                'twin_similarity': self.compute_similarity_matrices(twin_embeddings),
                'real_twin_similarity': self.embedding_extractor.compute_pairwise_similarity(
                    real_embeddings, twin_embeddings
                )
            }
        
        # Compute overall metrics
        overall_metrics = self.evaluator.compute_overall_metrics(all_metrics)
        
        # Generate visualizations
        self._generate_visualizations(all_similarity_matrices, output_dir)
        
        # Save results
        results = {
            'individual_metrics': all_metrics,
            'overall_metrics': overall_metrics,
            'similarity_matrices': all_similarity_matrices,
            'dataset_statistics': stats
        }
        
        self._save_results(results, output_dir)
        
        # Generate report
        self.evaluator.generate_evaluation_report(
            all_metrics, 
            os.path.join(output_dir, "evaluation_report.txt")
        )
        
        return results
    
    def evaluate_dataset(self, 
                        dataset_structure: Dict[str, Dict[str, List[str]]],
                        output_dir: str = "evaluation_results") -> Dict[str, any]:
        """
        Đánh giá toàn bộ dataset (legacy method)
        
        Args:
            dataset_structure: Dict với format phù hợp cho pipeline
            output_dir: Directory để lưu kết quả
            
        Returns:
            results: Dict chứa tất cả kết quả
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = []
        all_similarity_matrices = {}
        
        print(f"Evaluating dataset with {len(dataset_structure)} pairs...")
        
        for pair_id, pair_data in dataset_structure.items():
            print(f"\nProcessing {pair_id}...")
            
            # Load images
            real_images = self.load_and_preprocess_images(pair_data['real_images'])
            twin_images = self.load_and_preprocess_images(pair_data['twin_images'])
            
            if len(real_images) == 0 or len(twin_images) == 0:
                print(f"Warning: No images loaded for {pair_id}")
                continue
            
            # Evaluate pair
            metrics = self.evaluate_single_pair(real_images, twin_images, pair_id)
            all_metrics.append(metrics)
            
            # Save similarity matrices for visualization
            real_embeddings = self.extract_embeddings_optimized(real_images)
            twin_embeddings = self.extract_embeddings_optimized(twin_images)
            
            all_similarity_matrices[pair_id] = {
                'real_similarity': self.compute_similarity_matrices(real_embeddings),
                'twin_similarity': self.compute_similarity_matrices(twin_embeddings),
                'real_twin_similarity': self.embedding_extractor.compute_pairwise_similarity(
                    real_embeddings, twin_embeddings
                )
            }
        
        # Compute overall metrics
        overall_metrics = self.evaluator.compute_overall_metrics(all_metrics)
        
        # Generate visualizations
        self._generate_visualizations(all_similarity_matrices, output_dir)
        
        # Save results
        results = {
            'individual_metrics': all_metrics,
            'overall_metrics': overall_metrics,
            'similarity_matrices': all_similarity_matrices
        }
        
        self._save_results(results, output_dir)
        
        # Generate report
        self.evaluator.generate_evaluation_report(
            all_metrics, 
            os.path.join(output_dir, "evaluation_report.txt")
        )
        
        return results
    
    def _generate_visualizations(self, 
                                similarity_matrices: Dict[str, Dict[str, np.ndarray]],
                                output_dir: str):
        """
        Tạo visualizations cho similarity matrices
        """
        print("Generating visualizations...")
        
        for pair_id, matrices in similarity_matrices.items():
            pair_dir = os.path.join(output_dir, pair_id)
            os.makedirs(pair_dir, exist_ok=True)
            
            # Create heatmaps
            self.evaluator.create_similarity_heatmap(
                matrices['real_similarity'],
                f"Real Images Similarity - {pair_id}",
                os.path.join(pair_dir, "real_similarity.png")
            )
            
            self.evaluator.create_similarity_heatmap(
                matrices['twin_similarity'],
                f"Twin Images Similarity - {pair_id}",
                os.path.join(pair_dir, "twin_similarity.png")
            )
            
            # Use appropriate heatmap method based on matrix shape
            real_twin_matrix = matrices['real_twin_similarity']
            if real_twin_matrix.shape[0] != real_twin_matrix.shape[1]:
                # Unequal dimensions
                self.evaluator.create_unequal_similarity_heatmap(
                    real_twin_matrix,
                    f"Real vs Twin Similarity - {pair_id}",
                    os.path.join(pair_dir, "real_twin_similarity.png")
                )
            else:
                # Equal dimensions
                self.evaluator.create_similarity_heatmap(
                    real_twin_matrix,
                    f"Real vs Twin Similarity - {pair_id}",
                    os.path.join(pair_dir, "real_twin_similarity.png")
                )
    
    def _save_results(self, results: Dict[str, any], output_dir: str):
        """
        Lưu kết quả evaluation
        """
        # Save metrics as JSON
        metrics_data = {
            'individual_metrics': results['individual_metrics'],
            'overall_metrics': results['overall_metrics']
        }
        
        if 'dataset_statistics' in results:
            metrics_data['dataset_statistics'] = results['dataset_statistics']
        
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        # Save similarity matrices as numpy arrays
        matrices_dir = os.path.join(output_dir, "similarity_matrices")
        os.makedirs(matrices_dir, exist_ok=True)
        
        for pair_id, matrices in results['similarity_matrices'].items():
            pair_dir = os.path.join(matrices_dir, pair_id)
            os.makedirs(pair_dir, exist_ok=True)
            
            for matrix_name, matrix in matrices.items():
                np.save(
                    os.path.join(pair_dir, f"{matrix_name}.npy"),
                    matrix
                )
        
        print(f"Results saved to {output_dir}")


def create_dataset_structure_from_directories(real_dir: str, 
                                           twin_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Tạo dataset structure từ directories (legacy method)
    
    Args:
        real_dir: Directory chứa ảnh gốc
        twin_dir: Directory chứa ảnh song sinh
        
    Returns:
        dataset_structure: Dict với format phù hợp cho pipeline
    """
    dataset_structure = {}
    
    # Giả sử structure là:
    # real_dir/
    #   person_1/
    #     img1.jpg, img2.jpg, ...
    #   person_2/
    #     img1.jpg, img2.jpg, ...
    # twin_dir/
    #   person_1/
    #     img1_twin.jpg, img2_twin.jpg, ...
    #   person_2/
    #     img1_twin.jpg, img2_twin.jpg, ...
    
    real_persons = [d for d in os.listdir(real_dir) 
                   if os.path.isdir(os.path.join(real_dir, d))]
    
    for person in real_persons:
        real_person_dir = os.path.join(real_dir, person)
        twin_person_dir = os.path.join(twin_dir, person)
        
        if not os.path.exists(twin_person_dir):
            print(f"Warning: No twin directory for {person}")
            continue
        
        # Load image paths
        real_images = load_images_from_directory(real_person_dir)
        twin_images = load_images_from_directory(twin_person_dir)
        
        if len(real_images) > 0 and len(twin_images) > 0:
            dataset_structure[person] = {
                'real_images': real_images,
                'twin_images': twin_images
            }
    
    return dataset_structure


def run_evaluation_pipeline_from_json(id_images_path: str,
                                    pairs_twin_path: str,
                                    output_dir: str = "evaluation_results",
                                    model_name: str = "ir_50",
                                    device_ids: List[int] = None,
                                    batch_size: int = 32,
                                    balance_strategy: str = "min",
                                    handle_unequal: bool = True):
    """
    Chạy pipeline đánh giá từ JSON files
    
    Args:
        id_images_path: Path đến id_images.json
        pairs_twin_path: Path đến pairs_twin.json
        output_dir: Directory để lưu kết quả
        model_name: Tên model AdaFace
        device_ids: List GPU IDs
        batch_size: Kích thước batch
        balance_strategy: Chiến lược cân bằng
        handle_unequal: Có xử lý số lượng ảnh không đồng đều không
    """
    print("=== Twin Dataset Evaluation Pipeline (JSON Input) ===")
    print(f"ID images file: {id_images_path}")
    print(f"Pairs twin file: {pairs_twin_path}")
    print(f"Output directory: {output_dir}")
    print(f"Balance strategy: {balance_strategy}")
    print(f"Handle unequal counts: {handle_unequal}")
    
    # Validate input files
    if not os.path.exists(id_images_path):
        print(f"Error: ID images file {id_images_path} does not exist!")
        return
    
    if not os.path.exists(pairs_twin_path):
        print(f"Error: Pairs twin file {pairs_twin_path} does not exist!")
        return
    
    # Initialize pipeline
    pipeline = TwinDatasetEvaluationPipeline(
        model_name=model_name,
        device_ids=device_ids,
        batch_size=batch_size
    )
    
    # Run evaluation
    results = pipeline.evaluate_dataset_from_json(
        id_images_path, pairs_twin_path, output_dir, 
        balance_strategy, handle_unequal
    )
    
    # Print summary
    overall_metrics = results['overall_metrics']
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Overall Consistency Score: {overall_metrics['overall_consistency_mean']:.4f} ± {overall_metrics['overall_consistency_std']:.4f}")
    print(f"Overall Fidelity Score: {overall_metrics['overall_fidelity_mean']:.4f} ± {overall_metrics['overall_fidelity_std']:.4f}")
    print(f"Overall Score: {overall_metrics['overall_score']:.4f}")
    print(f"Worst Consistency Score: {overall_metrics['worst_consistency_score']:.4f}")
    print(f"Worst Fidelity Score: {overall_metrics['worst_fidelity_score']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def run_evaluation_pipeline(real_dir: str,
                          twin_dir: str,
                          output_dir: str = "evaluation_results",
                          model_name: str = "ir_50",
                          device_ids: List[int] = None,
                          batch_size: int = 32):
    """
    Chạy pipeline đánh giá hoàn chỉnh (legacy method)
    
    Args:
        real_dir: Directory chứa ảnh gốc
        twin_dir: Directory chứa ảnh song sinh
        output_dir: Directory để lưu kết quả
        model_name: Tên model AdaFace
        device_ids: List GPU IDs
        batch_size: Kích thước batch
    """
    print("=== Twin Dataset Evaluation Pipeline ===")
    print(f"Real images directory: {real_dir}")
    print(f"Twin images directory: {twin_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create dataset structure
    dataset_structure = create_dataset_structure_from_directories(real_dir, twin_dir)
    
    if len(dataset_structure) == 0:
        print("Error: No valid person directories found!")
        return
    
    print(f"Found {len(dataset_structure)} people to evaluate")
    
    # Initialize pipeline
    pipeline = TwinDatasetEvaluationPipeline(
        model_name=model_name,
        device_ids=device_ids,
        batch_size=batch_size
    )
    
    # Run evaluation
    results = pipeline.evaluate_dataset(dataset_structure, output_dir)
    
    # Print summary
    overall_metrics = results['overall_metrics']
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Overall Consistency Score: {overall_metrics['overall_consistency_mean']:.4f} ± {overall_metrics['overall_consistency_std']:.4f}")
    print(f"Overall Fidelity Score: {overall_metrics['overall_fidelity_mean']:.4f} ± {overall_metrics['overall_fidelity_std']:.4f}")
    print(f"Overall Score: {overall_metrics['overall_score']:.4f}")
    print(f"Worst Consistency Score: {overall_metrics['worst_consistency_score']:.4f}")
    print(f"Worst Fidelity Score: {overall_metrics['worst_fidelity_score']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results 