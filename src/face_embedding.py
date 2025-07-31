import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
import os
from tqdm import tqdm
import timm
from einops import rearrange


class AdaFaceEmbeddingExtractor:
    """
    Tối ưu hóa extractor embedding cho AdaFace với multi-GPU và batch processing
    """
    
    def __init__(self, model_name: str = "ir_50", device_ids: List[int] = None):
        """
        Args:
            model_name: Tên model AdaFace ('ir_50', 'ir_101', etc.)
            device_ids: List GPU IDs để sử dụng
        """
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.device = torch.device(f'cuda:{self.device_ids[0]}' if self.device_ids else 'cpu')
        
        # Load AdaFace model
        self.model = self._load_adaface_model(model_name)
        
        # Wrap model với DataParallel nếu có nhiều GPU
        if len(self.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing parameters
        self.input_size = 112
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
    
    def _load_adaface_model(self, model_name: str) -> nn.Module:
        """
        Load AdaFace model từ timm hoặc custom implementation
        """
        # Placeholder - bạn cần implement theo repo AdaFace
        # Có thể sử dụng timm hoặc load từ checkpoint
        model = timm.create_model('resnet50', num_classes=512, pretrained=False)
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess ảnh cho AdaFace model
        """
        # Resize và normalize
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        image = (image - np.array(self.mean)) / np.array(self.std)
        
        # Convert to tensor và add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image
    
    def extract_embeddings_batch(self, images: List[np.ndarray], 
                                batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings cho batch ảnh với tối ưu multi-GPU
        
        Args:
            images: List các ảnh numpy arrays
            batch_size: Kích thước batch
            
        Returns:
            embeddings: Array shape (N, embedding_dim)
        """
        embeddings = []
        
        # Process theo batches
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting embeddings"):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            # Stack thành batch tensor
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Tính similarity matrix cho tất cả embeddings một cách tối ưu
        
        Args:
            embeddings: Array shape (N, embedding_dim)
            
        Returns:
            similarity_matrix: Array shape (N, N)
        """
        # Convert to tensor và move to GPU
        embeddings_tensor = torch.from_numpy(embeddings).to(self.device)
        
        # Compute similarity matrix bằng matrix multiplication
        # S = E * E^T (cosine similarity với normalized vectors)
        with torch.no_grad():
            similarity_matrix = torch.mm(embeddings_tensor, embeddings_tensor.t())
        
        return similarity_matrix.cpu().numpy()
    
    def compute_pairwise_similarity(self, embeddings_1: np.ndarray, 
                                  embeddings_2: np.ndarray) -> np.ndarray:
        """
        Tính similarity matrix giữa 2 tập embeddings
        
        Args:
            embeddings_1: Array shape (N1, embedding_dim)
            embeddings_2: Array shape (N2, embedding_dim)
            
        Returns:
            similarity_matrix: Array shape (N1, N2)
        """
        # Convert to tensors
        emb1_tensor = torch.from_numpy(embeddings_1).to(self.device)
        emb2_tensor = torch.from_numpy(embeddings_2).to(self.device)
        
        # Compute similarity matrix
        with torch.no_grad():
            similarity_matrix = torch.mm(emb1_tensor, emb2_tensor.t())
        
        return similarity_matrix.cpu().numpy()


class FaceDataset:
    """
    Dataset class cho việc load và preprocess ảnh khuôn mặt
    """
    
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_path


def load_images_from_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Load tất cả ảnh từ directory
    
    Args:
        directory: Path đến directory chứa ảnh
        extensions: List các extension được chấp nhận
        
    Returns:
        image_paths: List các path đến ảnh
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    
    return image_paths 