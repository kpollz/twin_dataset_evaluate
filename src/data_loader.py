import json
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path


class TwinDatasetLoader:
    """
    Loader cho dữ liệu song sinh từ JSON files
    """
    
    def __init__(self, id_images_path: str, pairs_twin_path: str):
        """
        Args:
            id_images_path: Path đến file id_images.json
            pairs_twin_path: Path đến file pairs_twin.json
        """
        self.id_images_path = id_images_path
        self.pairs_twin_path = pairs_twin_path
        self.id_images_data = None
        self.pairs_twin_data = None
        
    def load_data(self) -> Tuple[Dict[str, List[str]], List[List[str]]]:
        """
        Load dữ liệu từ JSON files
        
        Returns:
            id_images_data: Dict mapping id -> list image paths
            pairs_twin_data: List các cặp song sinh
        """
        # Load id_images.json
        with open(self.id_images_path, 'r', encoding='utf-8') as f:
            self.id_images_data = json.load(f)
        
        # Load pairs_twin.json
        with open(self.pairs_twin_path, 'r', encoding='utf-8') as f:
            self.pairs_twin_data = json.load(f)
        
        print(f"Loaded {len(self.id_images_data)} IDs with images")
        print(f"Loaded {len(self.pairs_twin_data)} twin pairs")
        
        return self.id_images_data, self.pairs_twin_data
    
    def validate_data(self) -> bool:
        """
        Validate dữ liệu đã load
        
        Returns:
            is_valid: True nếu dữ liệu hợp lệ
        """
        if not self.id_images_data or not self.pairs_twin_data:
            print("Error: Data not loaded. Call load_data() first.")
            return False
        
        # Kiểm tra tất cả IDs trong pairs có tồn tại trong id_images
        all_ids_in_pairs = set()
        for pair in self.pairs_twin_data:
            if len(pair) != 2:
                print(f"Error: Invalid pair format: {pair}")
                return False
            all_ids_in_pairs.update(pair)
        
        missing_ids = all_ids_in_pairs - set(self.id_images_data.keys())
        if missing_ids:
            print(f"Error: Missing IDs in id_images.json: {missing_ids}")
            return False
        
        # Kiểm tra file ảnh có tồn tại
        missing_files = []
        for id_name, image_paths in self.id_images_data.items():
            for path in image_paths:
                if not os.path.exists(path):
                    missing_files.append(path)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} image files not found")
            print(f"First few missing files: {missing_files[:5]}")
        
        return True
    
    def get_twin_pairs_with_images(self) -> List[Tuple[str, List[str], str, List[str]]]:
        """
        Lấy danh sách các cặp song sinh với ảnh tương ứng
        
        Returns:
            List các tuple (id1, images1, id2, images2)
        """
        if not self.id_images_data or not self.pairs_twin_data:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        twin_pairs = []
        for pair in self.pairs_twin_data:
            id1, id2 = pair
            
            images1 = self.id_images_data[id1]
            images2 = self.id_images_data[id2]
            
            # Filter out non-existent files
            images1 = [img for img in images1 if os.path.exists(img)]
            images2 = [img for img in images2 if os.path.exists(img)]
            
            if len(images1) > 0 and len(images2) > 0:
                twin_pairs.append((id1, images1, id2, images2))
            else:
                print(f"Warning: Skipping pair {id1}-{id2} due to missing images")
        
        print(f"Found {len(twin_pairs)} valid twin pairs")
        return twin_pairs
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Lấy thống kê về dataset
        
        Returns:
            stats: Dict chứa thống kê
        """
        if not self.id_images_data or not self.pairs_twin_data:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Thống kê về số ảnh per ID
        images_per_id = {id_name: len(images) for id_name, images in self.id_images_data.items()}
        
        # Thống kê về pairs
        pair_stats = []
        for pair in self.pairs_twin_data:
            id1, id2 = pair
            num_images1 = len(self.id_images_data[id1])
            num_images2 = len(self.id_images_data[id2])
            pair_stats.append({
                'id1': id1,
                'id2': id2,
                'images1': num_images1,
                'images2': num_images2,
                'difference': abs(num_images1 - num_images2)
            })
        
        return {
            'total_ids': len(self.id_images_data),
            'total_pairs': len(self.pairs_twin_data),
            'images_per_id': images_per_id,
            'pair_statistics': pair_stats,
            'min_images_per_id': min(images_per_id.values()),
            'max_images_per_id': max(images_per_id.values()),
            'avg_images_per_id': np.mean(list(images_per_id.values())),
            'std_images_per_id': np.std(list(images_per_id.values()))
        }


def create_dataset_structure_from_json(id_images_path: str, 
                                     pairs_twin_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Tạo dataset structure từ JSON files cho pipeline
    
    Args:
        id_images_path: Path đến id_images.json
        pairs_twin_path: Path đến pairs_twin.json
        
    Returns:
        dataset_structure: Dict với format phù hợp cho pipeline
    """
    loader = TwinDatasetLoader(id_images_path, pairs_twin_path)
    loader.load_data()
    
    if not loader.validate_data():
        raise ValueError("Invalid data format")
    
    # Lấy twin pairs
    twin_pairs = loader.get_twin_pairs_with_images()
    
    # Tạo dataset structure
    dataset_structure = {}
    
    for i, (id1, images1, id2, images2) in enumerate(twin_pairs):
        # Tạo key cho pair này
        pair_key = f"pair_{i+1}_{id1}_{id2}"
        
        dataset_structure[pair_key] = {
            'real_images': images1,  # Ảnh của id1 (coi như real)
            'twin_images': images2,  # Ảnh của id2 (coi như twin)
            'id1': id1,
            'id2': id2,
            'num_images1': len(images1),
            'num_images2': len(images2)
        }
    
    return dataset_structure


def handle_unequal_image_counts(images1: List[str], 
                               images2: List[str],
                               strategy: str = "min") -> Tuple[List[str], List[str]]:
    """
    Xử lý trường hợp số lượng ảnh không đồng đều
    
    Args:
        images1: List ảnh của ID 1
        images2: List ảnh của ID 2
        strategy: Chiến lược xử lý ("min", "max", "random", "all")
        
    Returns:
        processed_images1, processed_images2: Hai list ảnh đã xử lý
    """
    len1, len2 = len(images1), len(images2)
    
    if len1 == len2:
        return images1, images2
    
    print(f"Unequal image counts: {len1} vs {len2}. Using strategy: {strategy}")
    
    if strategy == "min":
        # Lấy số lượng ảnh ít nhất
        min_count = min(len1, len2)
        return images1[:min_count], images2[:min_count]
    
    elif strategy == "max":
        # Lấy số lượng ảnh nhiều nhất, duplicate ảnh ít hơn
        max_count = max(len1, len2)
        if len1 < max_count:
            # Duplicate images1
            images1_extended = images1 * (max_count // len1) + images1[:max_count % len1]
            return images1_extended, images2
        else:
            # Duplicate images2
            images2_extended = images2 * (max_count // len2) + images2[:max_count % len2]
            return images1, images2_extended
    
    elif strategy == "random":
        # Random sampling để có số lượng bằng nhau
        min_count = min(len1, len2)
        indices1 = np.random.choice(len1, min_count, replace=False)
        indices2 = np.random.choice(len2, min_count, replace=False)
        return [images1[i] for i in indices1], [images2[i] for i in indices2]
    
    elif strategy == "all":
        # Giữ nguyên tất cả ảnh, xử lý trong metrics
        return images1, images2
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_balanced_dataset_structure(id_images_path: str,
                                    pairs_twin_path: str,
                                    balance_strategy: str = "min") -> Dict[str, Dict[str, List[str]]]:
    """
    Tạo dataset structure với xử lý cân bằng số lượng ảnh
    
    Args:
        id_images_path: Path đến id_images.json
        pairs_twin_path: Path đến pairs_twin.json
        balance_strategy: Chiến lược cân bằng ("min", "max", "random", "all")
        
    Returns:
        dataset_structure: Dict với format phù hợp cho pipeline
    """
    loader = TwinDatasetLoader(id_images_path, pairs_twin_path)
    loader.load_data()
    
    if not loader.validate_data():
        raise ValueError("Invalid data format")
    
    # Lấy twin pairs
    twin_pairs = loader.get_twin_pairs_with_images()
    
    # Tạo dataset structure với cân bằng
    dataset_structure = {}
    
    for i, (id1, images1, id2, images2) in enumerate(twin_pairs):
        # Xử lý số lượng ảnh không đồng đều
        balanced_images1, balanced_images2 = handle_unequal_image_counts(
            images1, images2, balance_strategy
        )
        
        # Tạo key cho pair này
        pair_key = f"pair_{i+1}_{id1}_{id2}"
        
        dataset_structure[pair_key] = {
            'real_images': balanced_images1,
            'twin_images': balanced_images2,
            'id1': id1,
            'id2': id2,
            'original_count1': len(images1),
            'original_count2': len(images2),
            'balanced_count': len(balanced_images1)
        }
    
    return dataset_structure 