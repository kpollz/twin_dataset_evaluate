# Báo cáo chi tiết phương pháp đánh giá một tập dữ liệu khuôn mặt song sinh tổng hợp bằng GenAI.

Hiện tại, các bộ dữ liệu khuôn mặt song sinh được tổng hợp bằng Generative AI đang được đánh giá bằng 2 yếu tố sau:
- **Consistency of Identity (Sự nhất quán về danh tính)**: Với N bức ảnh của người A, ta sinh ra được N bức ảnh khác tạm đặt tên là A', thì N bức ảnh mới đó phải thuộc cùng một người. Để làm được điều đó, ta dùng mô hình trích xuất đặc trưng và similarity score để vẽ lên một heatmap độ tương đồng giữa các bức ảnh này với nhau. Bằng cảm quan thì các giá trị trong heatmap càng gần 1 (cosine similarity) thì ta coi đó là cùng 1 người. Với M người thì ta sẽ có M heatmap.
- **Similarity to Real Images (Sự tương đồng với ảnh gốc)**: Tức là ngoài việc mong muốn N ảnh của A' là thuộc về cùng 1 người mà ta còn mong muốn N ảnh của A' phải giống hoặc gần giống với A. Tức là vẫn có những đặc điểm khác biệt nhỏ nhưng hệ thống Face Recognition khó nhận ra được. Để làm được điều này thì ta cũng vẽ heatmap như trên, chỉ là thay vì tính sự tương đồng giữa N ảnh của A' với nhau thì ta tính sự tương đồng giữa N ảnh của A' với N ảnh của A và ra được heatmap khác.

Vấn đề:
- Không có cơ sở lý luận hay định lượng nào để đánh giá mức độ *sáng* của heatmap là đủ 
- 2 yếu tố này đã thực sự đủ để đánh giá một bộ dữ liệu chưa ?
- Đánh giá thủ công và chủ quan cho M người. 
---

## 1. Số hóa Heatmap và Xác định "Ngưỡng Đủ Tốt"

Với vấn đề "heatmap sáng bao nhiêu là đủ", việc đánh giá bằng cảm quan không có tính nhân rộng và khách quan. Dưới đây là cách ta có thể số hóa các heatmap này thành những con số cụ thể.

Giả sử ta có ma trận tương đồng (heatmap) $H$ kích thước $N \times N$.

### **Đối với "Consistency of Identity" (Sự nhất quán về danh tính)**

Mục tiêu là các giá trị trong heatmap của các ảnh A' phải cao và đồng đều.

* **Đề xuất số hóa:** Thay vì nhìn cả heatmap, ta hãy tính 2 chỉ số sau cho mỗi người:
    1.  **Similarity Mean (Độ tương đồng trung bình):** Tính giá trị trung bình của tất cả các ô trong nửa trên (hoặc dưới) của ma trận tương đồng (không tính đường chéo). Giá trị này cho biết mức độ tương đồng trung bình trong nhóm ảnh A'.
        $$\text{Consistency Mean} = \text{mean}(H_{ij}) \quad \forall i < j$$
    2.  **Similarity Standard Deviation (Độ lệch chuẩn tương đồng):** Tính độ lệch chuẩn của các giá trị trên. Một độ lệch chuẩn **thấp** cho thấy các ảnh được sinh ra nhất quán về danh tính, không có ảnh nào "lạc loài".
        $$\text{Consistency Variance} = \text{std}(H_{ij}) \quad \forall i < j$$

* **Tạo một điểm số tổng hợp:** ta có thể kết hợp chúng thành một điểm duy nhất để dễ so sánh, ví dụ:
    $$\text{Consistency Score} = \text{Mean} - \text{Standard Deviation}$$
    Mục tiêu của ta là **tối đa hóa** điểm số này. Việc trừ đi độ lệch chuẩn sẽ "phạt" những trường hợp có độ tương đồng trung bình cao nhưng lại không đồng đều.

### **Đối với "Similarity to Real Images" (Sự tương đồng với ảnh gốc)**

Ở đây, heatmap của ta là ma trận tương đồng $N \times N$ giữa ảnh gốc A và ảnh sinh ra A'.

* **Đề xuất số hóa:**
    1.  **Fidelity Mean (Độ trung thực trung bình):** Tính giá trị trung bình của các ô trên đường chéo chính của heatmap, giả sử ảnh $A_i$ tương ứng với $A'_i$.
        $$\text{Fidelity Score} = \text{mean}(\text{diag}(H))$$
    2.  Hoặc, nếu không có sự tương ứng 1-1, ta có thể tính trung bình toàn bộ ma trận. Tuy nhiên, trung bình đường chéo thường phản ánh đúng hơn mục tiêu "giống với ảnh gốc tương ứng".

### **Vậy "Bao nhiêu là đủ?" 🎯**

Không có một con số vàng nào cho tất cả các bài toán. Ngưỡng này phụ thuộc vào mô hình Face Recognition ta đang sử dụng. Cách tốt nhất để xác định là **tạo một đường cơ sở (baseline)**:

1.  **Baseline "Cùng người":** Lấy N ảnh gốc của người A, tính heatmap tương đồng **giữa chúng**. Các giá trị trung bình và độ lệch chuẩn từ heatmap này chính là "ngưỡng vàng" mà bộ dữ liệu sinh ra của ta nên hướng tới. Dữ liệu A' của ta nên có `Consistency Score` gần với baseline này.
2.  **Baseline "Anh em sinh đôi thật":** Nếu ta có dữ liệu của các cặp song sinh thật, hãy tính độ tương đồng giữa họ. Đây là `Fidelity Score` lý tưởng mà ta muốn mô hình tạo sinh của mình đạt được.
3.  **Baseline "Khác người":** Tính độ tương đồng giữa người A và một người B hoàn toàn khác. Các giá trị `Fidelity Score` của cặp (A, A') phải **cao hơn đáng kể** so với baseline "khác người" này.


> ***Tóm lại, thay vì tìm một con số tuyệt đối, hãy so sánh các điểm số của dữ liệu sinh ra với các điểm số từ dữ liệu thật.***

---

## 2. Hai Yếu Tố Đánh Giá Đã Đủ Chưa?

Hai yếu tố của ta rất tốt nhưng mới chỉ tập trung vào khía cạnh **danh tính (identity)**. Một bộ dữ liệu tốt cần nhiều hơn thế. Dưới đây là các yếu tố quan trọng khác ta nên xem xét:

### **Realism/Image Quality (Tính chân thực / Chất lượng ảnh) 💡**

Ảnh A' có trông giống ảnh thật không hay có các chi tiết giả (artifacts)? Mắt người có thể nhận ra, nhưng để đo lường tự động, hãy dùng các metric tiêu chuẩn:

* **Fréchet Inception Distance (FID):** Đây là metric "tiêu chuẩn vàng" để đo khoảng cách phân phối giữa hai tập ảnh (ảnh thật và ảnh sinh ra). **FID càng thấp càng tốt**. Nó không chỉ so sánh pixel mà so sánh các đặc trưng bậc cao (deep features), phản ánh cả chất lượng và sự đa dạng.
* **Kernel Inception Distance (KID):** Tương tự FID nhưng ổn định hơn với số lượng mẫu nhỏ.

### **Diversity (Sự đa dạng) 🎭**

N ảnh A' của ta có đa dạng về góc mặt, biểu cảm, ánh sáng không? Hay chúng chỉ là những bản sao gần như y hệt nhau? Một bộ dữ liệu tốt cần sự đa dạng.

* **Cách đo lường:**
    * Tính trung bình độ lệch chuẩn của các vector đặc trưng của N ảnh A'. Độ lệch chuẩn lớn hơn thường cho thấy sự đa dạng cao hơn.
    * Sử dụng các metric như **Perceptual Path Length (PPL)** nếu ta có quyền truy cập vào không gian tiềm ẩn (latent space) của mô hình sinh.

### **Downstream Task Performance (Hiệu suất trên tác vụ chính)**

Hãy sử dụng bộ dữ liệu sinh ra của ta để tăng cường (augment) và huấn luyện lại (fine-tune) mô hình Face Recognition. Sau đó, đo lường hiệu suất của mô hình mới trên một tập kiểm thử (test set) chứa các cặp song sinh **mà mô hình chưa từng thấy**. Nếu độ chính xác tăng lên, bộ dữ liệu của ta thực sự hữu ích.

## 3. Đánh giá tổng quát thay vì đơn lẻ cho M người

Việc tổng hợp M giá trị thành một con số duy nhất cho toàn bộ dữ liệu là bước cuối cùng và rất quan trọng để có cái nhìn tổng quan. Dưới đây là các cách ta có thể thực hiện, từ đơn giản đến toàn diện.

### A. Phương pháp Trung bình cộng (Cách tiếp cận tiêu chuẩn)

Đây là cách đơn giản và phổ biến nhất. Đối với mỗi chỉ số đánh giá ta đã tính (ví dụ: `Consistency Score`, `Fidelity Score`), bạn chỉ cần lấy trung bình cộng của M giá trị tương ứng với M người.

* **`Overall Consistency Score`** = Trung bình (`Consistency Score` của người 1, người 2, ..., người M)
* **`Overall Fidelity Score`** = Trung bình (`Fidelity Score` của người 1, người 2, ..., người M)

**Ưu điểm:**
* Rất dễ tính toán và diễn giải.
* Cho một con số duy nhất đại diện cho hiệu suất trung bình của mô hình trên toàn bộ tập dữ liệu.

**Nhược điểm:**
* Một giá trị trung bình có thể che giấu các vấn đề. Ví dụ, mô hình có thể làm rất tốt với 90% số người nhưng lại thất bại hoàn toàn với 10% còn lại, và điểm trung bình vẫn có thể trông khá ổn.

Để khắc phục nhược điểm này, ta có thể tính thêm **Độ lệch chuẩn (Standard Deviation)** của M giá trị đó. Độ lệch chuẩn thấp cho thấy chất lượng sinh dữ liệu là đồng đều trên tất cả mọi người. Và tính một điểm tổng quát:

$$\text{Overall Score} = \text{Mean} - \text{Standard Deviation}$$


---

### B. Phân tích trường hợp tệ nhất (Kiểm tra độ bền)

Thay vì chỉ nhìn vào giá trị trung bình, hãy tìm giá trị **tệ nhất (thấp nhất)** trong M điểm số.

* **`Worst Consistency Score`** = Min(`Consistency Score` của người 1, người 2, ..., người M)

**Tại sao nó hữu ích?**
* Chỉ số này cho bạn biết "mắt xích yếu nhất" trong bộ dữ liệu là ở đâu. Trong nhận dạng khuôn mặt, việc thất bại dù chỉ trên một vài người cũng có thể là một vấn đề nghiêm trọng. Nếu điểm số tệ nhất này vẫn cao hơn một ngưỡng chấp nhận được, ta có thể tự tin hơn về độ tin cậy của mô hình.

---

### C. Tạo một "Điểm Chất Lượng Tổng Thể" duy nhất

Đây là cách tiếp cận cao cấp nhất, kết hợp tất cả các chỉ số thành **một con số cuối cùng** để đánh giá toàn bộ bộ dữ liệu.

**Bước 1: Chuẩn hóa các điểm số**

Mỗi chỉ số (Consistency, Fidelity, FID, Diversity...) có thang đo khác nhau. Ví dụ, FID càng thấp càng tốt, trong khi Fidelity càng cao càng tốt. Bạn cần đưa chúng về cùng một thang đo (ví dụ từ 0 đến 1), nơi 1 luôn là tốt nhất.
* **Với các chỉ số "càng cao càng tốt" (như Fidelity):** `Normalized_Score = Score` (nếu đã ở thang [0,1]) hoặc chuẩn hóa min-max.
* **Với các chỉ số "càng thấp càng tốt" (như FID):** `Normalized_Score = 1 / (1 + FID)` hoặc một phép biến đổi khác.

**Bước 2: Kết hợp thành điểm tổng thể bằng trọng số**

Ta cần quyết định yếu tố nào là quan trọng nhất và gán trọng số cho nó. Ví dụ, ta có thể cho rằng sự nhất quán về danh tính và độ chân thực là quan trọng nhất.

**`Overall Dataset Score`** =
$w_1 \times \text{Avg(Normalized Consistency)}$
\+ $w_2 \times \text{Avg(Normalized Fidelity)}$
\+ $w_3 \times \text{Normalized(FID)}$
\+ $w_4 \times \text{Avg(Normalized Diversity)}$
\+ ...

Trong đó, tổng các trọng số $w_1 + w_2 + w_3 + ... = 1$.

**Ưu điểm:**
* Cung cấp một con số duy nhất, rất tiện lợi để so sánh nhanh giữa các phiên bản mô hình tạo sinh khác nhau ("Mô hình A đạt 0.85 điểm, mô hình B đạt 0.92 điểm").

**Nhược điểm:**
* Việc chọn trọng số có thể mang tính chủ quan.


## 4. Tóm tắt

Để đánh giá bộ dữ liệu khuôn mặt song sinh một cách toàn diện và tự động, ta có một số metric sau:

| Yếu tố đánh giá | Metric đề xuất | Ý nghĩa |
| :--- | :--- | :--- |
| **1. Consistency (Nhất quán)** | `Consistency Score` (Mean - Std) của heatmap A' vs A' | Các ảnh A' phải thuộc cùng một người. **Càng cao càng tốt**. |
| **2. Fidelity (Trung thực)** | `Fidelity Score` (Mean) của heatmap A vs A' | Ảnh A' phải giống với ảnh gốc A. **Càng cao càng tốt**. |
| **3. Realism (Chân thực)** | `FID` / `KID` | Ảnh A' phải trông giống ảnh thật. **Càng thấp càng tốt**. |
| **4. Diversity (Đa dạng)** | Độ lệch chuẩn của các vector đặc trưng A' | Các ảnh A' phải đa dạng về biểu cảm, góc mặt. **Càng cao càng tốt** (trong giới hạn hợp lý). |
| **5. Usability (Hữu dụng)** | Độ chính xác của mô hình FR sau khi fine-tune | Phép thử cuối cùng: Dữ liệu có thực sự cải thiện hệ thống không? |
