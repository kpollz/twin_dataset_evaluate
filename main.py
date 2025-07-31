#!/usr/bin/env python3
"""
Script chính để chạy pipeline đánh giá bộ dữ liệu khuôn mặt song sinh
"""

import argparse
import os
import sys
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import run_evaluation_pipeline_from_json, run_evaluation_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Twin Dataset Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input type selection
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["json", "directory"],
        default="json",
        help="Loại input: json hoặc directory"
    )
    
    # JSON input arguments
    parser.add_argument(
        "--id_images_path",
        type=str,
        help="Path đến file id_images.json (chỉ dùng với --input_type json)"
    )
    
    parser.add_argument(
        "--pairs_twin_path", 
        type=str,
        help="Path đến file pairs_twin.json (chỉ dùng với --input_type json)"
    )
    
    # Directory input arguments (legacy)
    parser.add_argument(
        "--real_dir",
        type=str,
        help="Directory chứa ảnh gốc (chỉ dùng với --input_type directory)"
    )
    
    parser.add_argument(
        "--twin_dir", 
        type=str,
        help="Directory chứa ảnh song sinh (chỉ dùng với --input_type directory)"
    )
    
    # Common arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory để lưu kết quả"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="ir_50",
        help="Tên model AdaFace"
    )
    
    parser.add_argument(
        "--device_ids",
        type=int,
        nargs="+",
        default=None,
        help="List GPU IDs để sử dụng (mặc định: tất cả GPU có sẵn)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Kích thước batch cho processing"
    )
    
    # JSON-specific arguments
    parser.add_argument(
        "--balance_strategy",
        type=str,
        choices=["min", "max", "random", "all"],
        default="min",
        help="Chiến lược cân bằng số lượng ảnh (chỉ dùng với --input_type json)"
    )
    
    parser.add_argument(
        "--handle_unequal",
        action="store_true",
        default=True,
        help="Xử lý số lượng ảnh không đồng đều (chỉ dùng với --input_type json)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on input type
    if args.input_type == "json":
        if not args.id_images_path or not args.pairs_twin_path:
            print("Error: --id_images_path and --pairs_twin_path are required for JSON input!")
            return
        
        if not os.path.exists(args.id_images_path):
            print(f"Error: ID images file {args.id_images_path} does not exist!")
            return
        
        if not os.path.exists(args.pairs_twin_path):
            print(f"Error: Pairs twin file {args.pairs_twin_path} does not exist!")
            return
        
        # Run JSON-based evaluation
        try:
            results = run_evaluation_pipeline_from_json(
                id_images_path=args.id_images_path,
                pairs_twin_path=args.pairs_twin_path,
                output_dir=args.output_dir,
                model_name=args.model_name,
                device_ids=args.device_ids,
                batch_size=args.batch_size,
                balance_strategy=args.balance_strategy,
                handle_unequal=args.handle_unequal
            )
            
            if results:
                print("\n✅ Evaluation completed successfully!")
            else:
                print("\n❌ Evaluation failed!")
                
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.input_type == "directory":
        if not args.real_dir or not args.twin_dir:
            print("Error: --real_dir and --twin_dir are required for directory input!")
            return
        
        if not os.path.exists(args.real_dir):
            print(f"Error: Real directory {args.real_dir} does not exist!")
            return
        
        if not os.path.exists(args.twin_dir):
            print(f"Error: Twin directory {args.twin_dir} does not exist!")
            return
        
        # Run directory-based evaluation (legacy)
        try:
            results = run_evaluation_pipeline(
                real_dir=args.real_dir,
                twin_dir=args.twin_dir,
                output_dir=args.output_dir,
                model_name=args.model_name,
                device_ids=args.device_ids,
                batch_size=args.batch_size
            )
            
            if results:
                print("\n✅ Evaluation completed successfully!")
            else:
                print("\n❌ Evaluation failed!")
                
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 