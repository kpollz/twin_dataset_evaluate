# Pipeline Đánh Giá Bộ Dữ Liệu Khuôn Mặt Song Sinh

Pipeline tối ưu để đánh giá bộ dữ liệu khuôn mặt song sinh tổng hợp bằng Generative AI với multi-GPU và batch processing.

## 🚀 Tính Năng Chính

### 1. **Tối Ưu Multi-GPU**
- Hỗ trợ nhiều GPU đồng thời
- Tự động phân phối workload
- Tăng tốc độ xử lý lên đến N lần (N = số GPU)

### 2. **Batch Processing Tối Ưu**
- Xử lý ảnh theo batch để tận dụng GPU
- Tự động điều chỉnh batch size theo số GPU
- Giảm overhead của việc load/unload dữ liệu

### 3. **Matrix Computation Tối Ưu**
- Sử dụng matrix multiplication thay vì loop
- Tính similarity matrix một cách hiệu quả
- Tận dụng tối đa khả năng tính toán của GPU

### 4. **Metrics Toàn Diện**
- **Consistency Score**: Đánh giá sự nhất quán về danh tính
- **Fidelity Score**: Đánh giá độ tương đồng với ảnh gốc
- **Diversity Score**: Đánh giá sự đa dạng của ảnh sinh
- **Baseline Comparison**: So sánh với dữ liệu thật

## 📁 Cấu Trúc Dự Án

```
twin_dataset_evaluate/
├── src/
│   ├── __init__.py
│   ├── face_embedding.py      # AdaFace embedding extraction
│   ├── metrics.py             # Metrics calculation
│   └── pipeline.py            # Main evaluation pipeline
├── examples/
│   └── example_usage.py       # Usage examples
├── main.py                    # Command line interface
├── requirements.txt           # Dependencies
└── README_PIPELINE.md        # This file
```

## 🛠️ Cài Đặt

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Cài đặt AdaFace (nếu cần)
```bash
# Clone AdaFace repository
git clone https://github.com/mk-minchul/AdaFace.git
cd AdaFace
pip install -e .
```

## 📊 Cấu Trúc Dữ Liệu

Pipeline yêu cầu dữ liệu được tổ chức theo cấu trúc sau:

```
data/
├── real_images/
│   ├── person_1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── person_2/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── ...
└── twin_images/
    ├── person_1/
    │   ├── img1_twin.jpg
    │   ├── img2_twin.jpg
    │   └── ...
    ├── person_2/
    │   ├── img1_twin.jpg
    │   ├── img2_twin.jpg
    │   └── ...
    └── ...
```

## 🚀 Sử Dụng

### 1. Command Line Interface

```bash
python main.py \
    --real_dir data/real_images \
    --twin_dir data/twin_images \
    --output_dir results \
    --model_name ir_50 \
    --device_ids 0 1 2 3 \
    --batch_size 64
```

### 2. Python API

```python
from src.pipeline import run_evaluation_pipeline

# Chạy pipeline
results = run_evaluation_pipeline(
    real_dir="data/real_images",
    twin_dir="data/twin_images", 
    output_dir="results",
    model_name="ir_50",
    device_ids=[0, 1, 2, 3],
    batch_size=64
)

# In kết quả
print(f"Overall Score: {results['overall_metrics']['overall_score']:.4f}")
```

### 3. Sử dụng Pipeline Class

```python
from src.pipeline import TwinDatasetEvaluationPipeline

# Khởi tạo pipeline
pipeline = TwinDatasetEvaluationPipeline(
    model_name="ir_50",
    device_ids=[0, 1],
    batch_size=32
)

# Đánh giá dataset
results = pipeline.evaluate_dataset(dataset_structure, "output_dir")
```

## ⚡ Tối Ưu Hóa Hiệu Suất

### 1. **Multi-GPU Optimization**

```python
# Sử dụng tất cả GPU có sẵn
device_ids = list(range(torch.cuda.device_count()))

# Hoặc chỉ định GPU cụ thể
device_ids = [0, 1, 2, 3]
```

### 2. **Batch Size Optimization**

```python
# Batch size tối ưu theo số GPU
batch_size = 32 * len(device_ids)

# Hoặc điều chỉnh theo memory
batch_size = 64  # Cho GPU 8GB
batch_size = 128 # Cho GPU 16GB+
```

### 3. **Matrix Computation Tối Ưu**

Pipeline sử dụng matrix multiplication thay vì loop:

```python
# Thay vì loop qua từng cặp ảnh
for i in range(N):
    for j in range(N):
        similarity[i][j] = cosine_similarity(emb1[i], emb2[j])

# Sử dụng matrix multiplication
similarity_matrix = torch.mm(embeddings, embeddings.t())
```

## 📈 Metrics Được Tính

### 1. **Consistency Metrics**
- `consistency_mean`: Độ tương đồng trung bình giữa các ảnh A'
- `consistency_std`: Độ lệch chuẩn của độ tương đồng
- `consistency_score`: Điểm tổng hợp (mean - std)

### 2. **Fidelity Metrics**
- `fidelity_mean_diagonal`: Độ tương đồng trung bình 1-1
- `fidelity_mean_all`: Độ tương đồng trung bình toàn bộ
- `fidelity_score`: Điểm trung thực

### 3. **Diversity Metrics**
- `diversity_score`: Độ đa dạng của embeddings
- `distance_variance`: Phương sai của pairwise distances
- `num_unique_embeddings`: Số embeddings duy nhất

### 4. **Overall Metrics**
- `overall_consistency_mean/std`: Trung bình/độ lệch chuẩn consistency
- `overall_fidelity_mean/std`: Trung bình/độ lệch chuẩn fidelity
- `worst_consistency_score`: Điểm consistency thấp nhất
- `worst_fidelity_score`: Điểm fidelity thấp nhất
- `overall_score`: Điểm tổng hợp cuối cùng

## 📊 Kết Quả Đầu Ra

Pipeline tạo ra các file sau trong thư mục output:

```
results/
├── metrics.json                    # Metrics dạng JSON
├── evaluation_report.txt           # Báo cáo chi tiết
├── similarity_matrices/            # Ma trận similarity
│   ├── person_1/
│   │   ├── real_similarity.npy
│   │   ├── twin_similarity.npy
│   │   └── real_twin_similarity.npy
│   └── ...
└── person_1/                      # Heatmaps
    ├── real_similarity.png
    ├── twin_similarity.png
    └── real_twin_similarity.png
```

## 🔧 Tùy Chỉnh

### 1. **Thay Đổi Model**

```python
# Trong face_embedding.py, cập nhật _load_adaface_model
def _load_adaface_model(self, model_name: str) -> nn.Module:
    if model_name == "ir_50":
        # Load IR-50 model
        pass
    elif model_name == "ir_101":
        # Load IR-101 model
        pass
```

### 2. **Thêm Metrics Mới**

```python
# Trong metrics.py, thêm method mới
def compute_custom_metrics(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
    # Tính toán metric tùy chỉnh
    return {'custom_score': value}
```

### 3. **Tối Ưu Preprocessing**

```python
# Trong face_embedding.py, cập nhật preprocess_image
def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
    # Thêm augmentation, normalization tùy chỉnh
    pass
```

## 🐛 Troubleshooting

### 1. **Out of Memory**
```bash
# Giảm batch size
--batch_size 16

# Hoặc sử dụng ít GPU hơn
--device_ids 0 1
```

### 2. **Model Loading Error**
```bash
# Kiểm tra AdaFace installation
pip install -e /path/to/AdaFace

# Hoặc sử dụng model khác
--model_name ir_101
```

### 3. **Slow Performance**
```bash
# Tăng batch size
--batch_size 128

# Sử dụng nhiều GPU hơn
--device_ids 0 1 2 3 4 5 6 7
```

## 📝 Ví Dụ Sử Dụng

Xem file `examples/example_usage.py` để có các ví dụ chi tiết về:

- Sử dụng cơ bản
- Tối ưu embedding extraction
- Tính toán metrics
- Tối ưu batch processing
- Tối ưu multi-GPU

## 🤝 Đóng Góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.