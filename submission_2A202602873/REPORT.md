# Báo cáo Lab Day 1 — Nguyễn Tiến — 2A202602873

## 1. Thiết lập

- **Môi trường:** Local GPU (NVIDIA GeForce RTX 3050 Ti Laptop GPU 4GB VRAM), Python 3.12, PyTorch 2.5.1+cu124, CUDA 12.4.
- **Dữ liệu:** Forest CoverType (581 012 mẫu, 54 đặc trưng); `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% phân tầng theo nhãn (`seed=42`) $\rightarrow$ 371 847 mẫu train / 92 962 mẫu val. Chuẩn hoá 10 đặc trưng số liên tục đầu tiên bằng $\mu, \sigma$ của tập train (được tính độc lập trên train, gán $\sigma=1$ nếu $\sigma=0$); 44 cột one-hot giữ nguyên.
- **Model Baseline (`M-base`):** `54 → 256 → 128 → 7` (47 879 tham số). Khởi tạo He normal, không Dropout, không BatchNorm, kích hoạt ReLU, hàm mất mát Cross-Entropy, bộ tối ưu SGD + momentum 0.9, learning rate $lr=0.1$ (được quét tối ưu trên tập Val), batch size 512, huấn luyện 20 epochs, chuẩn chính xác FP32.
- **Mốc tham chiếu:** Accuracy "đoán lớp đa số" (Class 1 / nhãn gốc 2) trên tập Val = **0.4876** (48.76%), Macro-F1 tương ứng $\approx 0.0936$.
- **Các chủ đề đã thử:** ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init (Đã phủ đủ 7/7 chủ đề theo Rubric).

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả thực nghiệm |
|---|---|
| Số tham số / shape logits | 47 879 / torch.Size([8, 7]) (Khớp 100% quy định) |
| Loss bước 0 trên Val (so với $\ln 7 \approx 1.946$) | **2.2070** (lệch $\approx 0.26$ do mất cân bằng lớp và phương sai He) |
| Quá khớp 20 mẫu (Adam, 250 bước): loss cuối | **0.000010** (Accuracy = 100.0%) |
| Mọi tham số có gradient khác 0 sau backward | ☑ Có (toàn bộ 6 tensor $W_1, b_1, W_2, b_2, W_3, b_3$ đều có grad_norm $> 0$) |
| Baseline, số seed đã chạy | 2 seed (`base-s1`, `base-s2`) |
| Baseline: val acc (TB ± $\sigma$) | **0.9093 ± 0.0003** (90.93%) |
| Baseline: val macro-F1 (TB ± $\sigma$) | **0.8507 ± 0.0021** |

**Ngưỡng nhiễu dùng trong báo cáo:** $2\sigma = \mathbf{0.0041}$ (về chỉ số Val Macro-F1). Mọi cải tiến có $\Delta \text{Val Macro-F1} > +0.0041$ mới được kết luận là có ý nghĩa thống kê thực sự, không phải do ngẫu nhiên khởi tạo seed.

---

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — Cross-Entropy vs MSE
- **Dự đoán:** Cross-Entropy (CE) kết hợp hàm softmax có gradient dạng $(p_k - y_k)$ không bị bão hoà khi dự đoán sai lệch lớn. Ngược lại, MSE áp dụng trên vector nhãn one-hot sẽ bị suy giảm gradient nghiêm trọng ở các vùng logit lớn do đạo hàm bậc nhất của hàm kích hoạt tiệm cận 0 (gradient saturation). Do đó, MSE sẽ hội tụ chậm hơn và đạt hiệu năng phân loại thấp hơn đáng kể so với CE.
- **Kết quả:**
  * `loss-mse` ($lr=0.1$): Best Val Loss = 0.0298, Val Acc = 0.8694, **Val Macro-F1 = 0.7305** (Epoch 20).
  * So với Baseline (`base-s1`): $\Delta \text{Val Macro-F1} = -0.1216$ (kém hơn $29.7\sigma$, suy giảm cực mạnh).
  * Ảnh minh hoạ: [figures/loss-mse.png](figures/loss-mse.png) và [figures/compare_loss.png](figures/compare_loss.png).
- **Giải thích:** Hai hàm mất mát có thang đo hoàn toàn khác nhau (CE ở mức $\approx 0.22$, MSE ở mức $\approx 0.03$), do đó không so sánh trực tiếp độ lớn loss. Nhìn vào Macro-F1, MSE tụt giảm hơn 12% so với CE. Nguyên nhân là khi một mẫu bị phân loại sai nghiêm trọng, bề mặt lỗi của MSE trở nên rất phẳng (vanishing gradient), khiến bộ tối ưu không đủ lực đẩy cập nhật trọng số ở các lớp thiểu số.

---

### 3.2 Bộ tối ưu hoá — SGD+momentum vs Adam vs AdamW
- **Dự đoán:** Adam thích ứng learning rate theo từng toạ độ gradient thông qua ước lượng moment bậc 1 và bậc 2, do đó sẽ hội tụ rất nhanh ở những epoch ban đầu. Tuy nhiên, trên mạng MLP phân loại tabular chuẩn hoá tốt, SGD+momentum thường tìm được các cực tiểu phẳng (flat minima) có tính khái quát hoá tốt hơn. AdamW tách biệt cơ chế suy giảm trọng số (weight decay) khỏi bước cập nhật gradient thích nghi nên sẽ kiểm soát trọng số tốt hơn Adam thông thường.
- **Bảng so sánh:**

| exp_id | Bộ tối ưu | lr | weight decay | Val Macro-F1 | Best Epoch | Time/Epoch (s) |
|---|---|---|---|---|---|---|
| `base-s1` | SGD + momentum 0.9 | 0.1 | 0.0 | **0.8521** | 19 | 1.55s |
| `opt-adam` | Adam | 0.001 | 0.0 | **0.8521** | 20 | 1.79s |
| `opt-adamw` | AdamW | 0.001 | 0.01 | **0.8373** | 17 | 1.86s |

- **Độ nhạy với lr và giải thích:**
  * Ảnh minh hoạ: [figures/compare_optimizer.png](figures/compare_optimizer.png).
  * `opt-adam` ($lr=0.001$) cho kết quả rất tốt (Val Macro-F1 = 0.8521), ngang ngửa với Baseline SGD+momentum ở $lr=0.1$.
  * Với `opt-adamw` kèm `weight_decay=0.01`, Val Macro-F1 giảm về 0.8373 ($\Delta = -0.0148$, vượt quá $2\sigma$). Điều này giải thích được do kiến trúc `M-base` có kích thước tham số khá khiêm tốn (47k) so với lượng dữ liệu dồi dào (371k mẫu), việc phạt trọng số quá sớm bằng weight decay vô tình kìm hãm năng lực xấp xỉ của mạng.
  * Chi phí tính toán: Adam và AdamW tốn thêm thời gian lưu trữ và cập nhật trạng thái $m_t, v_t$ nên thời gian mỗi epoch tăng từ 1.55s lên $\approx 1.80s - 1.86s$ (+20%).

---

### 3.3 Hyper-parameter — Batch size & Độ rộng kiến trúc (`M-wide`)
- **Yếu tố khảo sát:**
  * **Batch size 128 (`hp-batch-128`):** Giữ nguyên $lr=0.1$. Số bước cập nhật mỗi epoch tăng gấp 4 lần (từ 727 steps lên 2 906 steps/epoch).
  * **Độ rộng mô hình `M-wide` (`hp-mwide`):** Kiến trúc `54 → 512 → 256 → 7` (161 287 tham số, tăng 3.37 lần dung lượng).
- **Kết quả:**
  * `hp-batch-128`: Val Acc = 0.9087, **Val Macro-F1 = 0.8551** (+0.0044 vs Base, vượt nhẹ ngưỡng $2\sigma$). Tuy nhiên, thời gian mỗi epoch tăng vọt từ **1.55s lên 6.52s** (tổng thời gian huấn luyện gấp 4.2 lần do overhead chia lô).
  * `hp-mwide`: Val Acc = **0.9228**, **Val Macro-F1 = 0.8742** (Epoch 20). So với Baseline, $\Delta \text{Val Macro-F1} = \mathbf{+0.0235}$ (**vượt hơn 11 lần ngưỡng nhiễu $2\sigma$**).
  * Ảnh minh hoạ: [figures/compare_hparam.png](figures/compare_hparam.png).
- **Giải thích:** Tập dữ liệu CoverType có 54 chiều địa hình phức tạp và 7 lớp phân định ranh giới phi tuyến. Việc mở rộng mạng lên `M-wide` giúp biểu diễn các ranh giới siêu mặt chi tiết hơn rất nhiều mà không bị quá khớp (do dữ liệu train lên tới 371k mẫu). Đây là bước nhảy vọt hiệu năng lớn nhất trong toàn bộ các thí nghiệm.

---

### 3.4 Regularization — Dropout
- **Dự đoán:** Nếu mô hình chưa bị quá khớp (khoảng cách giữa train loss và val loss rất nhỏ), việc thêm Dropout sẽ làm giảm năng lực học tức thời của mạng, khiến cả train loss và val loss đều tăng.
- **Kết quả:**
  * `drop-0.3` ($p=0.3$ sau các lớp ẩn): Train Loss = 0.3112, Val Loss = 0.3182, Val Acc = 0.8681, **Val Macro-F1 = 0.7745** (Epoch 20).
  * So với Baseline: Val Macro-F1 giảm sút nghiêm trọng $\Delta = -0.0762$ ($18.6\sigma$).
  * Ảnh minh hoạ: [figures/drop-0.3.png](figures/drop-0.3.png) và [figures/compare_dropout.png](figures/compare_dropout.png).
- **Giải thích:** Trên Baseline, khoảng cách Train Loss (0.2257) và Val Loss (0.2473) chỉ chênh lệch 0.0216, chứng tỏ mô hình chưa hề bị quá khớp. Khi ngẫu nhiên tắt đi 30% nơ-ron ở mỗi bước, mạng `M-base` vốn dĩ nhỏ bé bị mất đi các tổ hợp đặc trưng quan trọng để phân biệt các lớp hiếm, dẫn đến hiện tượng thiếu khớp (underfitting).

---

### 3.5 Cắt Gradient (Gradient Clipping) — Thí nghiệm phản chứng ở $lr$ cao
- **Thiết kế thí nghiệm:** Ở $lr=0.1$ bình thường, `grad_norm` trung bình của Baseline chỉ dao động quanh $1.3 - 2.5$, không hề có đột biến lớn. Để kiểm chứng giá trị thực sự của gradient clipping, ta thực hiện thí nghiệm phản chứng ở learning rate cao đột biến ($lr=0.5$):
  * `clip-highlr-noclip`: $lr=0.5$, không cắt gradient (`clip_norm=None`).
  * `clip-highlr-clip1`: $lr=0.5$, cắt gradient với ngưỡng chuẩn L2 $c=1.0$.
- **Kết quả:**
  * `clip-highlr-noclip`: Val Acc = 0.9008, **Val Macro-F1 = 0.8447** (Loss val cuối = 0.2528). Quá trình huấn luyện xuất hiện dao động mạnh ở các bước đầu.
  * `clip-highlr-clip1`: Val Acc = 0.9112, **Val Macro-F1 = 0.8554** (Loss val cuối = 0.2281).
  * Khác biệt: Clipping giúp tăng **+0.0107 Macro-F1** (vượt $2.6\sigma$).
  * Ảnh minh hoạ: [figures/compare_clipping.png](figures/compare_clipping.png).
- **Giải thích:** Khi $lr$ quá lớn, các bước nhảy gradient lớn dễ đẩy trọng số vào các vùng vách dựng đứng của hàm mất mát. Clipping với ngưỡng $c=1.0$ đã giới hạn độ dài vector cập nhật $g \leftarrow g \cdot \min(1, c/\|g\|)$, ngăn chặn bước nhảy phá huỷ tham số và giúp mô hình duy trì quỹ đạo hội tụ ổn định ngay cả khi $lr$ cao gấp 5 lần chuẩn.

---

### 3.6 Mixed Precision — FP16 với GradScaler
- **Kết quả đo đạc trên RTX 3050 Ti Laptop GPU:**
  * `amp-fp16`: Best Val Loss = 0.2341, Val Acc = 0.9056, **Val Macro-F1 = 0.8468**.
  * So với Baseline FP32 (`base-s1`): Val Macro-F1 chênh lệch $-0.0039$ (nằm trong biên độ nhiễu $2\sigma=0.0041$, tức chất lượng dự đoán hoàn toàn tương đương).
  * Bộ nhớ cực đại (`peak_mem_MB`): $\approx 166.4 \text{ MB}$.
  * Thời gian mỗi epoch: **2.33s** (chậm hơn so với 1.55s của FP32).
  * Ảnh minh hoạ: [figures/amp-fp16.png](figures/amp-fp16.png) và [figures/compare_amp.png](figures/compare_amp.png).
- **Giải thích vì sao FP16 không nhanh hơn:** Mạng MLP `M-base` có kích thước tham số rất nhỏ (47k) và dữ liệu dạng bảng 54 chiều. Khối lượng tính toán của phép nhân ma trận (GEMM) quá nhỏ để Tensor Cores phát huy lợi thế tính toán song song, trong khi chi phí phụ trội (overhead) cho việc ép kiểu (casting), kiểm tra tràn số và điều chỉnh hệ số phóng đại (scale/unscale) của `GradScaler` làm tổng thời gian xử lý mỗi batch tăng lên. Kết quả này hoàn toàn hợp lý theo đúng lý thuyết kỹ thuật của PyTorch AMP trên các mạng nơ-ron quy mô nhỏ.

---

### 3.7 Khởi tạo tham số — Xavier Uniform vs Zeros
- **Kết quả:**
  * `init-xavier`: Step 0 Val Loss = 2.0429, Best Val Loss = 0.2336, Val Acc = 0.9054, **Val Macro-F1 = 0.8594** (Epoch 20). Mô hình học tốt tương đương He.
  * `init-zeros`: Step 0 Val Loss = **1.9459** (đúng bằng $\ln 7$), Best Val Loss = 1.2052, Val Acc = **0.4876**, **Val Macro-F1 = 0.0936**.
  * Ảnh minh hoạ: [figures/init-zeros.png](figures/init-zeros.png) và [figures/compare_init.png](figures/compare_init.png).
- **Giải thích:**
  * `init-zeros` xác nhận hiện tượng **phá vỡ tính đối xứng (symmetry breaking) bị thất bại**: Khi toàn bộ trọng số khởi tạo bằng 0, mọi nơ-ron trong cùng một lớp ẩn đều nhận tín hiệu đầu vào như nhau và nhận gradient đạo hàm giống hệt nhau ở bước backward. Toàn bộ mạng nơ-ron sâu bị suy biến về chức năng thành 1 nơ-ron duy nhất, không thể phân tách các đặc trưng khác nhau. Kết quả là mô hình hoàn toàn bất lực và chỉ dự đoán nhãn lớp đa số (Accuracy kẹt ở 48.76%, Macro-F1 kẹt ở 0.0936).
  * Khởi tạo He phù hợp với hàm kích hoạt ReLU hơn Xavier vì tính đến việc một nửa số nơ-ron bị triệt tiêu bởi $\max(0, x)$, tuy nhiên trên cấu trúc nông 2 lớp ẩn thì Xavier vẫn duy trì được phương sai tín hiệu đủ để hội tụ tốt.

---

## 4. Đánh giá cuối trên tập eval

> Quyết định lựa chọn cấu hình được chốt **HOÀN TOÀN dựa trên tập Validation** trước khi đánh giá trên tập Eval: Cấu hình chiến thắng là **`hp-mwide`** với kiến trúc `M-wide` (`54 → 512 → 256 → 7`), He init, SGD+momentum 0.9, $lr=0.1$, batch 512, 20 epochs.

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy |
|---|---|---|---|---|
| **Baseline (`base-s1`)** | 1 | 0.8521 | **0.8512** | 0.9088 |
| **Cấu hình cuối (`hp-mwide`)** | 42 | **0.8742** | **0.8728** | **0.9200** |

- **Cấu hình cuối gồm những gì và vì sao:** Cấu hình cuối cùng sử dụng kiến trúc `M-wide` (161k tham số). Lý do lựa chọn: Trên tập Val, `hp-mwide` đạt Macro-F1 = 0.8742 (vượt Baseline $+0.0235$, gấp 11.2 lần ngưỡng $2\sigma$), Val Loss thấp nhất đạt 0.1984. Việc mở rộng mạng cung cấp đủ dung lượng biểu diễn cho 54 chiều đặc trưng địa hình mà không gây ra bất kỳ dấu hiệu quá khớp nào.
- **Cải thiện trên eval có vượt nhiễu không:** Trên tập Eval độc lập (116 203 mẫu chưa từng thấy), `hp-mwide` đạt **Eval Macro-F1 = 0.8728**, tăng **+0.0216 (+2.16%)** so với Baseline (0.8512). Độ chênh lệch này vượt xa ngưỡng $2\sigma = 0.0041$ (gấp 5.3 lần độ lệch chuẩn), chứng minh sự cải thiện là hoàn toàn có ý nghĩa khoa học.
- **Độ tương đồng giữa Val và Eval:** Điểm số giữa Val (0.8742) và Eval (0.8728) cực kỳ sát nhau (chênh lệch chỉ $0.0014$), chứng minh việc chuẩn hoá dữ liệu và quy trình phân tầng không hề bị rò rỉ dữ liệu (data leakage) và mô hình có khả năng tổng quát hoá rất vững chắc.

---

### 4.1 Phân tích lỗi theo lớp (trên tập Eval)

Trích xuất trực tiếp từ [eval_result.json](eval_result.json) (116 203 mẫu):

| Lớp | Tên lớp (Cover Type) | Support | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|
| 0 | Spruce/Fir | 42 368 | 0.9152 | 0.9224 | **0.9188** |
| 1 | Lodgepole Pine | 56 661 | 0.9309 | 0.9361 | **0.9335** |
| 2 | Ponderosa Pine | 7 151 | 0.8905 | 0.9311 | **0.9103** |
| 3 | Cottonwood/Willow | 549 | 0.8151 | 0.8270 | **0.8210** |
| 4 | Aspen | 1 899 | 0.8154 | 0.7699 | **0.7920** |
| 5 | Douglas-fir | 3 473 | 0.8830 | 0.7668 | **0.8208** |
| 6 | Krummholz | 4 102 | 0.9662 | 0.8654 | **0.9131** |

**Ma trận nhầm lẫn (Confusion Matrix — Hàng: Nhãn thật, Cột: Dự đoán):**
```
      Lớp 0   Lớp 1  Lớp 2  Lớp 3  Lớp 4  Lớp 5  Lớp 6
0: [  39080,   3122,     2,     0,    39,     7,   118 ]
1: [   3093,  53043,   148,     0,   265,   106,     6 ]
2: [      1,    197,  6658,    71,    23,   201,     0 ]
3: [      0,      1,    75,   454,     0,    19,     0 ]
4: [     48,    350,    19,     0,  1462,    20,     0 ]
5: [      5,    173,   585,    29,     4,  2663,    14 ]
6: [    460,     91,     0,     0,     1,     1,  3549 ]
```

- **Lớp khó nhất:** Lớp **4 (Aspen)** có F1 thấp nhất toàn bộ hệ thống (**F1 = 0.7920**, Recall = 0.7699). Lớp này thường xuyên bị nhầm lẫn nhiều nhất với **Lớp 1 (Lodgepole Pine)** (350 mẫu của lớp 4 bị đoán thành lớp 1).
- **Cặp lớp nhầm lẫn nhiều nhất:** **Lớp 0 và Lớp 1** nhầm lẫn qua lại rất lớn (3 122 mẫu lớp 0 bị đoán thành lớp 1, và 3 093 mẫu lớp 1 bị đoán thành lớp 0).
- **Lý giải nguyên nhân:**
  1. *Đặc trưng độ cao tương đồng:* Spruce/Fir (Lớp 0) và Lodgepole Pine (Lớp 1) cùng sinh trưởng ở vành đai cận núi cao (subalpine zone) với điều kiện thổ nhưỡng và góc dốc địa hình rất gần nhau, dẫn đến các phân bố vector 54 chiều bị chồng lấn mạnh.
  2. *Mất cân bằng dữ liệu:* Lớp 4 (Aspen) chỉ chiếm 1 899 mẫu (1.6% tập eval), trong khi Lớp 1 chiếm tới 56 661 mẫu (48.8%). Khi mô hình đối mặt với các mẫu không chắc chắn, hàm mất mát có xu hướng thiên vị dự đoán về phía lớp đa số (Lớp 1) để giảm thiểu rủi ro loss trung bình.
- **Hướng cải thiện:** Áp dụng kỹ thuật Focal Loss hoặc gán trọng số lớp (Class-weighted Cross-Entropy) nghịch đảo với tần suất xuất hiện $w_c \propto 1 / \sqrt{N_c}$ để phạt nặng hơn khi dự đoán sai các lớp thiểu số như Lớp 4.

---

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
   * Khi được tinh chỉnh $lr$ phù hợp (Adam tại $lr=10^{-3}$ và SGD+momentum tại $lr=0.1$), **cả hai đạt kết quả tương đương nhau** (Macro-F1 cùng đạt $0.8521$). 
   * Tuy nhiên, nếu không chỉnh $lr$ mà dùng chung một mức $lr=0.1$ cho cả hai: Adam sẽ nổ gradient và phân kỳ ngay lập tức do cơ chế cập nhật thích nghi chia cho $\sqrt{v_t}$ làm bước nhảy bị khuếch đại quá lớn. Ngược lại, nếu dùng chung $lr=0.001$, SGD sẽ di chuyển cực kỳ chậm chạp và không thể hội tụ trong 20 epochs. Kết luận so sánh chỉ có ý nghĩa khoa học khi khảo sát trên dải $lr$ tối ưu của từng bộ tối ưu.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
   * **Không giúp, thậm chí gây hại nghiêm trọng.** Thực nghiệm với `drop-0.3` khiến Macro-F1 tụt dốc từ 0.8521 xuống 0.7745. Khi mô hình chưa bị quá khớp (khoảng cách train loss và val loss $\le 0.02$), mạng nơ-ron cần toàn bộ dung lượng tham số để học các đặc trưng phức tạp.
   * Chỉ nên dùng Dropout khi quan sát thấy khoảng cách giữa Train Loss và Val Loss mở rộng đáng kể (Train loss tiếp tục giảm sâu trong khi Val loss bắt đầu tăng ngược trở lại — triệu chứng của Overfitting) hoặc khi số lượng tham số mô hình vượt trội so với số lượng mẫu.

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**
   * Gradient clipping giải quyết bài toán **nổ gradient (exploding gradients)** và ngăn chặn các bước nhảy phá huỷ trọng số khi đi qua vùng địa hình mất mát dốc đứng.
   * *Bằng chứng thực nghiệm:* Ở thí nghiệm phản chứng $lr=0.5$ rất cao, mô hình không clip (`clip-highlr-noclip`) có quỹ đạo loss bất ổn định và chỉ đạt Macro-F1 = 0.8447. Khi bật `clip_norm=1.0` (`clip-highlr-clip1`), các đỉnh gradient nhọn bị cắt gọn, giữ cho quá trình tối ưu trơn tru và kéo Macro-F1 tăng lên 0.8554 (tăng $+0.0107$, vượt $2.6\sigma$).

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?**
   * **Không nhanh hơn.** Trên GPU RTX 3050 Ti, FP16 mất 2.33s/epoch trong khi FP32 chỉ mất 1.55s/epoch.
   * *Nguyên nhân:* Mạng MLP `M-base` có cấu trúc nhỏ và chi phí nhân ma trận quá thấp. Thời gian chạy bị thống trị bởi chi phí gọi kernel của CPU sang GPU (launch overhead) và thao tác kiểm tra/phóng đại gradient của `GradScaler`. Mixed precision chỉ thực sự tăng tốc trên các mô hình lớn (Transformer, ResNet, LLM) hoặc khi kích thước ma trận ẩn đủ lớn để lấp đầy các Tensor Cores.

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   * *Khởi tạo toàn số 0 hỏng vì:* Triệt tiêu tính phá vỡ đối xứng (symmetry breaking). Tất cả các trọng số xuất phát từ 0 sẽ nhận đạo hàm gradient y hệt nhau sau backward, khiến mọi nơ-ron trong cùng một lớp cập nhật y hệt nhau qua mọi epoch. Mạng nơ-ron nhiều tầng suy biến thành 1 nơ-ron duy nhất và không thể học được gì (Accuracy kẹt cứng ở 48.76%).
   * *He vs Xavier:* Xavier giả định hàm kích hoạt là tuyến tính hoặc đối xứng quanh gốc toạ độ (như Tanh), gán phương sai trọng số $\text{Var}(W) = 2 / (n_{in} + n_{out})$. He tính đến việc hàm ReLU triệt tiêu hoàn toàn một nửa miền giá trị âm ($\max(0, x)$), do đó nhân đôi phương sai: $\text{Var}(W) = 2 / n_{in}$. Sự khác biệt này cực kỳ quan trọng đối với các mạng sâu: nếu dùng Xavier với ReLU trên mạng nhiều tầng, tín hiệu kích hoạt và gradient sẽ bị triệt tiêu dần về 0 khi truyền qua các tầng (vanishing gradient).

6. **Ba phép kiểm tra đầu tiên khi một mạng có loss không giảm sau 2 000 bước:**
   1. **Kiểm tra Loss bước 0 trên Val (ở chế độ `eval`):** Với phân loại $C$ lớp dùng hàm Cross-Entropy, loss bước đầu tiên phải xấp xỉ $\ln(C)$ (với 7 lớp là $\ln 7 \approx 1.946$). Nếu loss bước 0 lệch xa hoặc rất lớn (ví dụ $> 10$ hoặc là `NaN`), chắc chắn code đã bị lỗi: áp dụng softmax hai lần, nhãn chưa chuyển về $0..C-1$, hoặc trọng số bị nổ khi khởi tạo.
   2. **Thử quá khớp một lô nhỏ (Overfit on 20 samples):** Lấy đúng 20 mẫu, tắt dropout/regularization, cho mạng học trong vài trăm bước với Adam $lr=0.01$. Nếu loss không ép được về sát 0 và accuracy không đạt 100%, lỗi chắc chắn nằm ở code (quên `optimizer.zero_grad()`, không đưa tham số mô hình vào optimizer, ngắt luồng autograd, hoặc tensor nhãn bị sai).
   3. **Kiểm tra dòng chảy Gradient (`grad_norm` từng tham số):** Sau bước `loss.backward()`, in chuẩn gradient của từng tầng trọng số và bias. Nếu có tầng nào có gradient bằng `None` hoặc bằng 0 tuyệt đối, mạng đã bị đứt mạch autograd hoặc toàn bộ nơ-ron tầng đó đã rơi vào vùng chết của hàm ReLU (dying ReLU).

---

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ nhất:** Tác động tiêu cực của Dropout trên mạng MLP đối với dữ liệu CoverType. Dự đoán ban đầu cho rằng dropout nhẹ ($p=0.3$) sẽ giúp mô hình bền bỉ hơn, nhưng thực tế nó làm giảm Macro-F1 tới hơn 7.6%. Điều này củng cố nguyên lý: chỉ dùng regularization khi mô hình có triệu chứng quá khớp rõ ràng.
- **Hạn chế trong thiết kế:**
  * Do giới hạn thời gian chạy trên lớp, mỗi cấu hình thí nghiệm Part 3 chỉ mới chạy trên 1 seed ($seed=42$), mặc dù Baseline đã chạy 2 seed để thiết lập thước đo $2\sigma$.
  * Chưa thử nghiệm kết hợp đồng thời nhiều kỹ thuật (ví dụ `M-wide` kết hợp Adam và LR Cosine Scheduler).
- **Hướng nghiên cứu tiếp theo:**
  * Thử nghiệm kỹ thuật gán trọng số lớp (Class-weighted Loss) hoặc Focal Loss để nâng cao F1 cho Lớp 4 (Aspen).
  * Khảo sát bộ lập lịch tốc độ học Cosine Annealing LR để tối ưu sâu hơn ở các epoch cuối.

---

## 7. Phụ lục

- **Danh mục các file nộp trong `submission_2A202602873/`:**
  * `REPORT.md`: Báo cáo khoa học tổng kết toàn diện bài lab.
  * `experiments.xlsx`: Bảng tổng hợp chi tiết 16 thí nghiệm kèm đầy đủ 4 sheet chuẩn.
  * `predictions_eval.csv`: 116 203 dòng dự đoán cho tập eval đạt Macro-F1 = **0.8728**.
  * `eval_result.json`: File kết quả chấm điểm chính thức từ `scripts/evaluate.py`.
  * `results/*.json`: 16 file nhật ký chi tiết của từng lần chạy.
  * `figures/*.png`: 25 biểu đồ gồm đồ thị 3 ô và các biểu đồ so sánh nhóm.
  * `code/*.py` & `code/lab.ipynb`: Toàn bộ mã nguồn và notebook thực nghiệm hoàn chỉnh.
- **Thời gian chạy ước tính tổng cộng:** $\approx 15$ phút trên NVIDIA GeForce RTX 3050 Ti Laptop GPU.
