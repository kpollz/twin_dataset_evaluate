import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import cv2
from scipy.stats import pearsonr


class TwinDatasetEvaluator:
    """
    Evaluator cho bộ dữ liệu khuôn mặt song sinh tổng hợp
    """
    
    def __init__(self):
        self.metrics = {}
    
    def compute_consistency_metrics(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Tính metrics cho Consistency of Identity
        
        Args:
            similarity_matrix: Ma trận similarity NxN của các ảnh A'
            
        Returns:
            metrics: Dict chứa các metrics
        """
        # Lấy upper triangle (không tính diagonal)
        upper_triangle = np.triu(similarity_matrix, k=1)
        values = upper_triangle[upper_triangle != 0]
        
        if len(values) == 0:
            return {
                'consistency_mean': 0.0,
                'consistency_std': 0.0,
                'consistency_score': 0.0
            }
        
        consistency_mean = np.mean(values)
        consistency_std = np.std(values)
        consistency_score = consistency_mean - consistency_std
        
        return {
            'consistency_mean': float(consistency_mean),
            'consistency_std': float(consistency_std),
            'consistency_score': float(consistency_score)
        }
    
    def compute_fidelity_metrics(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Tính metrics cho Similarity to Real Images (trường hợp ma trận vuông)
        
        Args:
            similarity_matrix: Ma trận similarity NxN giữa ảnh gốc A và ảnh sinh A'
            
        Returns:
            metrics: Dict chứa các metrics
        """
        # Tính mean của diagonal (tương ứng 1-1)
        diagonal_values = np.diag(similarity_matrix)
        fidelity_mean = np.mean(diagonal_values)
        
        # Hoặc tính mean toàn bộ matrix nếu không có tương ứng 1-1
        fidelity_mean_all = np.mean(similarity_matrix)
        
        return {
            'fidelity_mean_diagonal': float(fidelity_mean),
            'fidelity_mean_all': float(fidelity_mean_all),
            'fidelity_score': float(fidelity_mean)  # Sử dụng diagonal mean
        }
    
    def compute_fidelity_metrics_unequal_counts(self, 
                                              embeddings1: np.ndarray, 
                                              embeddings2: np.ndarray) -> Dict[str, float]:
        """
        Tính fidelity metrics cho trường hợp số lượng ảnh không đồng đều
        Không cần cân bằng, tính trực tiếp từ ma trận N1×N2
        
        Args:
            embeddings1: Embeddings của ảnh gốc (N1, dim)
            embeddings2: Embeddings của ảnh song sinh (N2, dim)
            
        Returns:
            metrics: Dict chứa các metrics
        """
        # Tính pairwise similarity matrix N1×N2
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        # Tính các metrics khác nhau
        metrics = {}
        
        # 1. Mean similarity toàn bộ
        metrics['fidelity_mean_all'] = float(np.mean(similarity_matrix))
        
        # 2. Mean similarity cho từng ảnh gốc (max similarity với ảnh song sinh)
        max_similarities_per_real = np.max(similarity_matrix, axis=1)
        metrics['fidelity_mean_max_per_real'] = float(np.mean(max_similarities_per_real))
        
        # 3. Mean similarity cho từng ảnh song sinh (max similarity với ảnh gốc)
        max_similarities_per_twin = np.max(similarity_matrix, axis=0)
        metrics['fidelity_mean_max_per_twin'] = float(np.mean(max_similarities_per_twin))
        
        # 4. Top-k similarity cho mỗi ảnh gốc
        k_per_real = min(3, embeddings2.shape[0])  # Top-3 hoặc tất cả nếu ít hơn 3
        top_k_similarities_per_real = []
        for i in range(embeddings1.shape[0]):
            similarities = similarity_matrix[i]
            top_k_indices = np.argsort(similarities)[-k_per_real:]
            top_k_similarities_per_real.extend(similarities[top_k_indices])
        
        metrics['fidelity_mean_top_k_per_real'] = float(np.mean(top_k_similarities_per_real))
        
        # 5. Top-k similarity cho mỗi ảnh song sinh
        k_per_twin = min(3, embeddings1.shape[0])  # Top-3 hoặc tất cả nếu ít hơn 3
        top_k_similarities_per_twin = []
        for j in range(embeddings2.shape[0]):
            similarities = similarity_matrix[:, j]
            top_k_indices = np.argsort(similarities)[-k_per_twin:]
            top_k_similarities_per_twin.extend(similarities[top_k_indices])
        
        metrics['fidelity_mean_top_k_per_twin'] = float(np.mean(top_k_similarities_per_twin))
        
        # 6. Fidelity score chính (sử dụng mean max per real)
        metrics['fidelity_score'] = metrics['fidelity_mean_max_per_real']
        
        # 7. Thêm thông tin về kích thước ma trận
        metrics['matrix_shape'] = f"{embeddings1.shape[0]}x{embeddings2.shape[0]}"
        metrics['num_real_images'] = embeddings1.shape[0]
        metrics['num_twin_images'] = embeddings2.shape[0]
        
        return metrics
    
    def compute_diversity_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Tính metrics cho Diversity
        
        Args:
            embeddings: Array shape (N, embedding_dim)
            
        Returns:
            metrics: Dict chứa các metrics
        """
        # Tính độ lệch chuẩn của embeddings
        embedding_std = np.std(embeddings, axis=0)
        diversity_score = np.mean(embedding_std)
        
        # Tính variance của pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        
        distance_variance = np.var(distances) if distances else 0.0
        
        return {
            'diversity_score': float(diversity_score),
            'distance_variance': float(distance_variance),
            'num_unique_embeddings': len(np.unique(embeddings, axis=0))
        }
    
    def compute_baseline_metrics(self, real_similarity_matrix: np.ndarray,
                               twin_similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Tính baseline metrics để so sánh (trường hợp ma trận vuông)
        
        Args:
            real_similarity_matrix: Similarity matrix của ảnh thật
            twin_similarity_matrix: Similarity matrix của ảnh song sinh
            
        Returns:
            metrics: Dict chứa baseline metrics
        """
        # Tính metrics cho real images
        real_metrics = self.compute_consistency_metrics(real_similarity_matrix)
        
        # Tính metrics cho twin images
        twin_metrics = self.compute_consistency_metrics(twin_similarity_matrix)
        
        # So sánh với baseline
        baseline_gap = real_metrics['consistency_score'] - twin_metrics['consistency_score']
        
        return {
            'real_consistency_score': real_metrics['consistency_score'],
            'twin_consistency_score': twin_metrics['consistency_score'],
            'baseline_gap': float(baseline_gap),
            'relative_performance': float(twin_metrics['consistency_score'] / real_metrics['consistency_score'])
        }
    
    def compute_baseline_metrics_unequal_counts(self, 
                                              real_embeddings: np.ndarray,
                                              twin_embeddings: np.ndarray) -> Dict[str, float]:
        """
        Tính baseline metrics cho trường hợp số lượng ảnh không đồng đều
        Không cần cân bằng, tính trực tiếp
        
        Args:
            real_embeddings: Embeddings của ảnh thật
            twin_embeddings: Embeddings của ảnh song sinh
            
        Returns:
            metrics: Dict chứa baseline metrics
        """
        # Tính consistency cho real images (ma trận vuông)
        real_similarity = np.dot(real_embeddings, real_embeddings.T)
        real_consistency = self.compute_consistency_metrics(real_similarity)
        
        # Tính consistency cho twin images (ma trận vuông)
        twin_similarity = np.dot(twin_embeddings, twin_embeddings.T)
        twin_consistency = self.compute_consistency_metrics(twin_similarity)
        
        # Tính fidelity (ma trận không vuông)
        fidelity_metrics = self.compute_fidelity_metrics_unequal_counts(
            real_embeddings, twin_embeddings
        )
        
        # So sánh với baseline
        baseline_gap = real_consistency['consistency_score'] - twin_consistency['consistency_score']
        
        return {
            'real_consistency_score': real_consistency['consistency_score'],
            'twin_consistency_score': twin_consistency['consistency_score'],
            'baseline_gap': float(baseline_gap),
            'relative_performance': float(twin_consistency['consistency_score'] / real_consistency['consistency_score']),
            **fidelity_metrics
        }
    
    def compute_overall_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Tính overall metrics cho toàn bộ dataset
        
        Args:
            all_metrics: List các metrics cho từng người
            
        Returns:
            overall_metrics: Dict chứa overall metrics
        """
        # Tính mean và std của các metrics
        consistency_scores = [m.get('consistency_score', 0) for m in all_metrics]
        fidelity_scores = [m.get('fidelity_score', 0) for m in all_metrics]
        
        overall_consistency_mean = np.mean(consistency_scores)
        overall_consistency_std = np.std(consistency_scores)
        overall_fidelity_mean = np.mean(fidelity_scores)
        overall_fidelity_std = np.std(fidelity_scores)
        
        # Worst case analysis
        worst_consistency = np.min(consistency_scores)
        worst_fidelity = np.min(fidelity_scores)
        
        # Overall score với penalty cho variance
        overall_score = (overall_consistency_mean + overall_fidelity_mean) / 2 - (overall_consistency_std + overall_fidelity_std) / 2
        
        return {
            'overall_consistency_mean': float(overall_consistency_mean),
            'overall_consistency_std': float(overall_consistency_std),
            'overall_fidelity_mean': float(overall_fidelity_mean),
            'overall_fidelity_std': float(overall_fidelity_std),
            'worst_consistency_score': float(worst_consistency),
            'worst_fidelity_score': float(worst_fidelity),
            'overall_score': float(overall_score)
        }
    
    def create_similarity_heatmap(self, similarity_matrix: np.ndarray, 
                                 title: str = "Similarity Matrix",
                                 save_path: Optional[str] = None) -> None:
        """
        Tạo heatmap cho similarity matrix vuông
        
        Args:
            similarity_matrix: Ma trận similarity vuông
            title: Tiêu đề cho heatmap
            save_path: Path để lưu heatmap
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   vmin=0, 
                   vmax=1,
                   square=True)
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_unequal_similarity_heatmap(self, 
                                        similarity_matrix: np.ndarray,
                                        title: str = "Similarity Matrix",
                                        save_path: Optional[str] = None) -> None:
        """
        Tạo heatmap cho similarity matrix không vuông (N1×N2)
        
        Args:
            similarity_matrix: Ma trận similarity (N1, N2)
            title: Tiêu đề cho heatmap
            save_path: Path để lưu heatmap
        """
        plt.figure(figsize=(12, 8))
        
        # Tạo heatmap với kích thước phù hợp
        sns.heatmap(similarity_matrix, 
                   annot=False,  # Không annotate vì có thể quá nhiều
                   fmt='.3f',
                   cmap='viridis',
                   vmin=0, 
                   vmax=1,
                   square=False,  # Không square vì kích thước khác nhau
                   cbar_kws={'label': 'Similarity Score'})
        
        plt.title(title)
        plt.xlabel("Twin Images")
        plt.ylabel("Real Images")
        
        # Thêm thông tin về kích thước
        n_real, n_twin = similarity_matrix.shape
        plt.text(0.02, 0.98, f'Matrix: {n_real}×{n_twin}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, real_metrics: Dict[str, float],
                               twin_metrics: Dict[str, float],
                               save_path: Optional[str] = None) -> None:
        """
        Vẽ biểu đồ so sánh metrics giữa real và twin images
        
        Args:
            real_metrics: Metrics của ảnh thật
            twin_metrics: Metrics của ảnh song sinh
            save_path: Path để lưu biểu đồ
        """
        metrics_names = ['Consistency Score', 'Fidelity Score']
        real_values = [real_metrics.get('consistency_score', 0), 
                      real_metrics.get('fidelity_score', 0)]
        twin_values = [twin_metrics.get('consistency_score', 0), 
                      twin_metrics.get('fidelity_score', 0)]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, real_values, width, label='Real Images', alpha=0.8)
        plt.bar(x + width/2, twin_values, width, label='Twin Images', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Comparison of Real vs Twin Images')
        plt.xticks(x, metrics_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, all_metrics: List[Dict[str, float]],
                                 output_path: str = "evaluation_report.txt") -> None:
        """
        Tạo báo cáo đánh giá chi tiết
        
        Args:
            all_metrics: List các metrics cho từng người
            output_path: Path để lưu báo cáo
        """
        overall_metrics = self.compute_overall_metrics(all_metrics)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== BÁO CÁO ĐÁNH GIÁ BỘ DỮ LIỆU SONG SINH ===\n\n")
            
            f.write("1. OVERALL METRICS:\n")
            f.write(f"   - Overall Consistency Score: {overall_metrics['overall_consistency_mean']:.4f} ± {overall_metrics['overall_consistency_std']:.4f}\n")
            f.write(f"   - Overall Fidelity Score: {overall_metrics['overall_fidelity_mean']:.4f} ± {overall_metrics['overall_fidelity_std']:.4f}\n")
            f.write(f"   - Overall Score: {overall_metrics['overall_score']:.4f}\n")
            f.write(f"   - Worst Consistency Score: {overall_metrics['worst_consistency_score']:.4f}\n")
            f.write(f"   - Worst Fidelity Score: {overall_metrics['worst_fidelity_score']:.4f}\n\n")
            
            f.write("2. INDIVIDUAL METRICS:\n")
            for i, metrics in enumerate(all_metrics):
                f.write(f"   Pair {i+1}:\n")
                f.write(f"     - Consistency Score: {metrics.get('consistency_score', 0):.4f}\n")
                f.write(f"     - Fidelity Score: {metrics.get('fidelity_score', 0):.4f}\n")
                if 'diversity_score' in metrics:
                    f.write(f"     - Diversity Score: {metrics['diversity_score']:.4f}\n")
                if 'matrix_shape' in metrics:
                    f.write(f"     - Matrix Shape: {metrics['matrix_shape']}\n")
                    f.write(f"     - Real Images: {metrics.get('num_real_images', 'N/A')}\n")
                    f.write(f"     - Twin Images: {metrics.get('num_twin_images', 'N/A')}\n")
                if 'original_count1' in metrics:
                    f.write(f"     - Original Counts: {metrics['original_count1']} vs {metrics['original_count2']}\n")
                f.write("\n")
        
        print(f"Evaluation report saved to: {output_path}")


def calculate_fid_score(real_images: List[np.ndarray], 
                       generated_images: List[np.ndarray]) -> float:
    """
    Tính FID score (Fréchet Inception Distance)
    
    Args:
        real_images: List ảnh thật
        generated_images: List ảnh được sinh ra
        
    Returns:
        fid_score: FID score (càng thấp càng tốt)
    """
    # Placeholder implementation
    # Bạn cần implement FID calculation hoặc sử dụng thư viện có sẵn
    # Ví dụ: pytorch-fid, clean-fid
    
    # Giả sử tính khoảng cách trung bình giữa embeddings
    real_embeddings = np.random.rand(len(real_images), 512)  # Placeholder
    gen_embeddings = np.random.rand(len(generated_images), 512)  # Placeholder
    
    # Tính FID score
    real_mean = np.mean(real_embeddings, axis=0)
    gen_mean = np.mean(gen_embeddings, axis=0)
    
    real_cov = np.cov(real_embeddings.T)
    gen_cov = np.cov(gen_embeddings.T)
    
    # FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))
    mean_diff = real_mean - gen_mean
    fid_score = np.dot(mean_diff, mean_diff)
    
    # Add covariance term (simplified)
    cov_diff = real_cov + gen_cov
    fid_score += np.trace(cov_diff)
    
    return float(fid_score) 