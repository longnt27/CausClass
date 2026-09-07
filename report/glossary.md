# Glossary: Thuật ngữ dùng trong báo cáo

Bảng thuật ngữ này dùng để đảm bảo cách viết nhất quán trong toàn bộ báo cáo.

## 1. Thuật ngữ hệ thống

| Thuật ngữ | Cách dùng khuyến nghị | Ghi chú |
|---|---|---|
| Raw classroom video | Video lớp học đầu vào | Có thể dùng "video lớp học" trong văn bản chính. |
| Framework | Framework hoặc khung phương pháp | Nếu muốn thuần Việt, dùng "khung phương pháp". |
| Pipeline | Pipeline hoặc quy trình xử lý | Có thể dùng "pipeline" nếu đã quen trong ngữ cảnh kỹ thuật. |
| Stage | Giai đoạn | Trong hình vẽ có thể giữ "Stage" để gọn. Trong văn bản nên dùng "giai đoạn". |
| Observation report | Báo cáo quan sát | Không gọi là báo cáo đánh giá chất lượng giảng dạy. |
| Evidence-based observation report | Báo cáo quan sát dựa trên bằng chứng | Tên tốt cho đầu ra cuối. |

## 2. Thuật ngữ Stage 1

| Thuật ngữ | Cách dùng khuyến nghị | Ghi chú |
|---|---|---|
| Visual perception | Nhận thức thị giác | |
| Temporal abstraction | Trừu tượng hóa thời gian | Dùng khi nói về việc chuyển sự kiện theo khung hình thành chuỗi thời gian. |
| Frame extraction | Trích xuất khung hình | Có thể viết "trích xuất hoặc lấy mẫu khung hình". |
| Sampling | Lấy mẫu | Dùng nếu không xử lý mọi khung hình. |
| Detection | Phát hiện | Dùng cho YOLO. |
| Object tracking | Theo dõi đối tượng | Dùng cho ByteTrack. |
| ByteTrack | ByteTrack | Giữ nguyên tên thuật toán. |
| Behavior event | Sự kiện hành vi | Một lần phát hiện hành vi tại một thời điểm hoặc trong một khoảng ngắn. |
| Behavior counting | Đếm hành vi | Dùng cho bước tổng hợp số lần xuất hiện hành vi. |
| Temporal binning | Gom theo khoảng thời gian | Có thể ghi "temporal binning" trong ngoặc ở lần đầu. |
| Time bin | Khoảng thời gian hoặc bin thời gian | Nên dùng "khoảng thời gian" trong văn bản chính. |
| Behavior share | Tỉ lệ hành vi | Là tỉ lệ sự kiện hành vi trong một khoảng thời gian. |
| Normalization | Chuẩn hóa | Dùng cho bước chuyển số đếm thành tỉ lệ. |
| Multivariate time-series matrix | Ma trận chuỗi thời gian đa biến | Thuật ngữ quan trọng, cần dùng nhất quán. |

## 3. Thuật ngữ mô hình YOLO và MHSA

| Thuật ngữ | Cách dùng khuyến nghị | Ghi chú |
|---|---|---|
| YOLO26n | YOLO26n | Giữ nguyên tên mô hình. |
| YOLO26n-MHSA | YOLO26n-MHSA | Tên mô hình đề xuất trong Stage 1. |
| YOLOv8-MHSA | YOLOv8-MHSA | Dùng khi nhắc công trình liên quan đã truyền cảm hứng. |
| Multi-Head Self-Attention | Cơ chế tự chú ý đa đầu | Lần đầu có thể viết "cơ chế tự chú ý đa đầu (Multi-Head Self-Attention, MHSA)". |
| MHSA | MHSA | Sau lần định nghĩa đầu tiên, dùng MHSA. |
| Backbone | Backbone hoặc mạng trích xuất đặc trưng | Nếu cần học thuật hơn, dùng "mạng trích xuất đặc trưng". |
| Head | Head phát hiện | Có thể giữ "head" nếu đang mô tả kiến trúc YOLO. |
| Feature map | Bản đồ đặc trưng | Dùng trong phần giải thích MHSA. |
| Long-range spatial dependency | Phụ thuộc không gian dài | Dùng để giải thích lợi ích kỳ vọng của MHSA. |
| Fine-tuning | Tinh chỉnh | Nên dùng "tinh chỉnh mô hình". |
| Ablation study | Nghiên cứu cắt lớp hoặc ablation study | Trong bảng có thể viết "ablation study". Văn bản nên giải thích là so sánh có hoặc không có một thành phần. |

## 4. Thuật ngữ nhãn hành vi

| Nhãn | Cách ghi trong báo cáo | Lưu ý diễn giải |
|---|---|---|
| Read | Read hoặc đọc | Nếu bảng dùng tên nhãn gốc, giữ Read. |
| Write | Write hoặc viết | Nếu bảng dùng tên nhãn gốc, giữ Write. |
| Phone | Phone hoặc sử dụng điện thoại | Không tự kết luận là dùng điện thoại sai mục đích. |
| Bow | Bow hoặc cúi đầu | Nhãn mơ hồ, có thể gồm đọc, viết, nhìn xuống bàn hoặc dùng điện thoại. |
| Lean | Lean hoặc nằm ra bàn |  |
| Talk | Talk hoặc nói chuyện | Không tự kết luận là nói chuyện riêng hoặc mất tập trung. |
| Hand | Hand hoặc giơ tay | Chỉ xuất hiện trong một số bộ dữ liệu, không thuộc năm nhãn chung. |

## 5. Thuật ngữ Stage 2

| Thuật ngữ | Cách dùng khuyến nghị | Ghi chú |
|---|---|---|
| Temporal association | Liên kết thời gian | Thuật ngữ chính nên dùng thay cho "nhân quả". |
| Lagged dependency | Phụ thuộc trễ | Dùng khi nhấn mạnh quan hệ giữa thời điểm trước và sau. |
| Temporal association graph | Đồ thị liên kết thời gian | Thuật ngữ chính cho đồ thị đầu ra. |
| Behavior dynamics graph | Đồ thị động lực hành vi | Có thể dùng khi nói tổng quát hơn. |
| Correlation graph | Đồ thị tương quan | Không ưu tiên vì "correlation" thường gợi ý quan hệ không hướng. |
| Causal graph | Đồ thị nhân quả | Chỉ dùng khi thảo luận nền tảng hoặc giới hạn, không dùng làm tên chính. |
| AERCA | AERCA | Giữ nguyên tên phương pháp. |
| Baseline AERCA | AERCA cơ sở | Dùng cho phương pháp không có LLM-guided refinement. |
| Masked AERCA | Masked AERCA | Có thể dịch là "AERCA có mặt nạ", nhưng giữ nguyên sẽ tự nhiên hơn. |
| Graph edit | Chỉnh sửa đồ thị | Gồm thêm cạnh, xóa cạnh, đảo chiều cạnh. |
| Candidate graph | Đồ thị ứng viên | Đồ thị được tạo ra sau một hoặc nhiều chỉnh sửa. |
| Candidate mask | Mặt nạ ứng viên | Mặt nạ dùng để ràng buộc hoặc kiểm chứng chỉnh sửa. |
| Local re-initialization | Khởi tạo lại cục bộ | Dùng trong bước chạy masked AERCA sau chỉnh sửa. |
| Verifier | Bộ kiểm chứng | Trong hệ thống, AERCA và hàm đánh giá đóng vai trò kiểm chứng. |
| Proposal generator | Bộ đề xuất | LLM đóng vai trò bộ đề xuất chỉnh sửa đồ thị. |
| Beam search | Tìm kiếm chùm hoặc beam search | Có thể giữ "beam search" sau khi đã giải thích. |
| Greedy search | Tìm kiếm tham lam hoặc greedy search | Có thể giữ "greedy" trong tên phương pháp. |
| Random beam | Beam ngẫu nhiên | Baseline dùng chỉnh sửa ngẫu nhiên. |
| LLM greedy | LLM greedy | Phương pháp dùng LLM nhưng chỉ giữ một nhánh tốt nhất. |
| LLM beam | LLM beam | Phương pháp đề xuất chính. |
| Search memory | Bộ nhớ tìm kiếm | Lưu lịch sử, đồ thị đã thử, chỉnh sửa đã bị loại. |
| Tabu list | Danh sách cấm hoặc tabu list | Nếu code có dùng cơ chế tránh lặp, có thể dùng thuật ngữ này. |

## 6. Thuật ngữ đánh giá phát hiện đối tượng

| Thuật ngữ | Cách dùng khuyến nghị | Ghi chú |
|---|---|---|
| Precision | Precision | Có thể giải thích là tỉ lệ phát hiện đúng trong các phát hiện được mô hình đưa ra. |
| Recall | Recall | Có thể giải thích là tỉ lệ đối tượng đúng được mô hình phát hiện. |
| F1 | F1 | Trung bình điều hòa giữa Precision và Recall. |
| mAP@50 | mAP@50 | Chỉ số chính cho phát hiện đối tượng ở ngưỡng IoU 0.5. |
| mAP@50-95 | mAP@50-95 | Chỉ số nghiêm ngặt hơn, lấy trung bình trên nhiều ngưỡng IoU. |
| IoU | IoU | Intersection over Union, có thể giải thích ở chương nền tảng. |
| Confusion matrix | Ma trận nhầm lẫn | Dùng nếu có phân tích nhầm nhãn. |

## 7. Thuật ngữ đánh giá đồ thị

| Thuật ngữ | Cách dùng khuyến nghị | Ghi chú |
|---|---|---|
| Graph recovery | Khôi phục cấu trúc đồ thị | Dùng trong dữ liệu tổng hợp có ground truth. |
| Ground truth graph | Đồ thị ground truth | Có thể viết "đồ thị ground truth" hoặc "đồ thị chuẩn". |
| Edge | Cạnh | Liên kết có hướng giữa hai hành vi. |
| Directed edge | Cạnh có hướng | Ví dụ Phone -> Read. |
| Edge count | Số cạnh | Dùng để đánh giá độ thưa hoặc độ dày của đồ thị. |
| False positive edge | Cạnh dương tính giả | Cạnh được dự đoán nhưng không có trong ground truth. |
| False negative edge | Cạnh âm tính giả | Cạnh có trong ground truth nhưng không được dự đoán. |
| Structural precision | Precision cấu trúc | Precision tính trên cạnh đồ thị. |
| Structural recall | Recall cấu trúc | Recall tính trên cạnh đồ thị. |
| Structural F1 | F1 cấu trúc | F1 tính trên cạnh đồ thị. |
| BIC | BIC | Bayesian Information Criterion, dùng như chỉ số chẩn đoán. |
| MSE | MSE | Mean Squared Error, dùng như chỉ số lỗi dự báo. |

## 8. Quy tắc diễn giải cạnh trong đồ thị

Nếu có cạnh:

\[
A \rightarrow B
\]

Nên diễn giải:

> Hành vi A ở các khoảng thời gian trước có liên hệ dự báo với sự thay đổi của hành vi B ở các khoảng thời gian sau.

Không nên diễn giải:

> Hành vi A gây ra hành vi B.

Nếu trọng số cạnh dương:

> Khi A tăng, B có xu hướng tăng trong các khoảng thời gian tiếp theo, theo mô hình đã học.

Nếu trọng số cạnh âm:

> Khi A tăng, B có xu hướng giảm trong các khoảng thời gian tiếp theo, theo mô hình đã học.

Cần thêm câu thận trọng:

> Quan hệ này là liên kết thời gian trên dữ liệu quan sát, không phải bằng chứng nhân quả thực nghiệm.

## 9. Tên phương pháp trong bảng kết quả

| Tên trong code | Tên nên dùng trong báo cáo |
|---|---|
| aerca_baseline | AERCA cơ sở |
| random_beam | Beam ngẫu nhiên |
| llm_greedy | LLM greedy |
| llm_beam | LLM beam hoặc LLM beam đề xuất |

Nếu muốn nhấn mạnh phương pháp chính:

> LLM beam (đề xuất)

## 10. Cách dịch các cụm thường gặp

| Cụm tiếng Anh | Cụm tiếng Việt nên dùng |
|---|---|
| end-to-end pipeline | pipeline đầu cuối |
| classroom behavior dynamics | động lực hành vi trong lớp học |
| temporal pattern | mẫu hình thời gian |
| behavioral signal | tín hiệu hành vi |
| behavior statistics | thống kê hành vi |
| trend summary | tóm tắt xu hướng |
| evidence summary | tóm tắt bằng chứng |
| qualitative case study | nghiên cứu trường hợp định tính |
| synthetic test suite | bộ kiểm thử tổng hợp |
| hold-out validation | kiểm chứng trên tập giữ lại |
| proposal strategy | chiến lược đề xuất |
| search prior | prior tìm kiếm |
| semantic prior | prior ngữ nghĩa |
| verification objective | hàm mục tiêu kiểm chứng |
| local minimum | cực tiểu cục bộ |
| sparse graph | đồ thị thưa |
| dense graph | đồ thị dày |

## 11. Các cụm nên tránh

| Không nên dùng | Lý do | Thay bằng |
|---|---|---|
| Quan hệ nhân quả chắc chắn | Quá mạnh | Liên kết thời gian |
| Học sinh mất tập trung | Thiếu ngữ cảnh | Tín hiệu hành vi cần kiểm tra |
| Điện thoại gây giảm đọc | Suy diễn nhân quả | Phone có liên kết âm với Read |
| LLM phát hiện đồ thị | LLM không phải bộ kiểm chứng | LLM đề xuất chỉnh sửa đồ thị |
| AERCA chứng minh quan hệ | Quá mạnh | AERCA kiểm chứng ứng viên theo hàm mục tiêu |
| Báo cáo đưa ra lời khuyên | Dễ bị hỏi cách đánh giá | Báo cáo hỗ trợ quan sát và kiểm tra thủ công |

## 12. Ký hiệu toán học đề xuất

| Ký hiệu | Ý nghĩa |
|---|---|
| \(X \in \mathbb{R}^{T \times K}\) | Ma trận chuỗi thời gian đa biến |
| \(T\) | Số khoảng thời gian |
| \(K\) | Số nhãn hành vi |
| \(x_{t,k}\) | Tỉ lệ hành vi \(k\) tại khoảng thời gian \(t\) |
| \(G\) | Đồ thị liên kết thời gian |
| \(e_{i \rightarrow j}\) | Cạnh có hướng từ hành vi \(i\) đến hành vi \(j\) |
| \(w_{i,j}\) | Trọng số cạnh từ \(i\) đến \(j\) |
| \(B\) | Beam hoặc tập đồ thị ứng viên |
| \(k\) | Beam width hoặc số ứng viên được giữ lại |

Công thức gợi ý cho Stage 1:

\[
x_{t,k} = \frac{c_{t,k}}{\sum_{j=1}^{K} c_{t,j}}
\]

trong đó \(c_{t,k}\) là số sự kiện hành vi \(k\) trong khoảng thời gian \(t\).

## 13. Câu mẫu dùng trong báo cáo

### Mô tả Stage 1

> Giai đoạn 1 chuyển video lớp học thành ma trận chuỗi thời gian đa biến. Mỗi phần tử của ma trận biểu diễn tỉ lệ của một hành vi trong một khoảng thời gian cố định.

### Mô tả Stage 2

> Giai đoạn 2 sử dụng AERCA để khởi tạo đồ thị liên kết thời gian, sau đó dùng LLM để đề xuất các chỉnh sửa đồ thị. Mỗi chỉnh sửa được kiểm chứng bằng masked AERCA trước khi được giữ lại trong quá trình tìm kiếm.

### Mô tả LLM beam

> LLM beam đạt kết quả tốt nhất trong các phương pháp được so sánh, với F1 trung bình 0.5043 và precision trung bình 0.4952 trên 100 bộ kiểm thử tổng hợp. Kết quả này cho thấy beam search giúp cải thiện khả năng chọn lọc cạnh so với AERCA cơ sở và LLM greedy.

### Mô tả giới hạn

> Do dữ liệu lớp học thật không có đồ thị ground truth, kết quả trên video thật được trình bày như một nghiên cứu trường hợp định tính. Các cạnh trong đồ thị cần được diễn giải như liên kết thời gian cần kiểm tra, không phải kết luận nhân quả tuyệt đối.
