# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):468, cab, 1, 0.510915, ImageNet-1K
- Record này mô tả toàn ảnh như thế nào?
Record này mô tả toàn bộ ảnh là một hình ảnh thuộc lớp traffic.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Class list được định nghĩa bởi dataset và cấu hình taxonomy khi checkpoint được huấn luyện. Với yolo11n-cls.pt trong bài này, checkpoint dự đoán các lớp thuộc taxonomy ImageNet-1K.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
class_id là mã số để máy đối chiếu chính xác; class_name giúp con người hiểu lớp được dự đoán; taxonomy_name cho biết lớp đó thuộc hệ phân loại nào, tránh nhầm khi các hệ khác nhau có thể dùng tên lớp tương tự.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Guideline phải quy định ảnh được gán một lớp cho chủ thể chính hay nhiều lớp cho tất cả chủ thể. Đồng thời cần nêu cách xử lý chủ thể phụ, chủ thể bị che khuất/cắt mép và trường hợp không đủ thông tin cần chuyển reviewer quyết định.
- Vì sao model score không phải ground truth?
Model score chỉ là mức độ tin cậy do mô hình ước lượng cho một prediction. Score cao không đảm bảo dự đoán đúng. Ground truth phải do annotator hoặc reviewer xác định dựa trên guideline, không được lấy trực tiếp từ score của model.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
Chọn record person: score = 0.912625, bbox_xyxy = [385.33, 69.24, 498.92, 348.92], bbox_width = 113.58 pixel, bbox_height = 279.68 pixel.
- Diễn giải vị trí box bằng lời:
Box bao quanh một người đứng ở khu vực giữa và lệch phải ảnh kitchen. Box bắt đầu tại khoảng (385.33, 69.24) và kết thúc tại (498.92, 348.92) trên ảnh kích thước 640×427 pixel.
- So sánh số prediction ở hai threshold:
Ở threshold 0.35 có 11 prediction. Khi tăng threshold lên 0.60, còn 6 prediction.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Threshold 0.35 giữ lại nhiều object hơn, nên độ bao phủ cao hơn nhưng reviewer phải kiểm tra nhiều prediction hơn, trong đó có thể có prediction chưa chắc chắn. Threshold 0.60 giảm số prediction cần xem, nhưng có nguy cơ bỏ sót object nhỏ hoặc có score thấp.
- Đề xuất một quy tắc box chặt:
Vẽ box nhỏ nhất bao trọn phần object nhìn thấy, không lấy quá nhiều nền xung quanh; dùng đúng định dạng [x_min, y_min, x_max, y_max] và bảo đảm box nằm trong giới hạn ảnh.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quy định có gán nhãn object bị che khuất hoặc cắt khỏi mép ảnh hay không, box bao phần nhìn thấy hay toàn bộ object ước lượng, và mức che khuất nào được chấp nhận. Nếu không xác định rõ object hoặc vị trí box, annotator cần chuyển reviewer quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box?
- `instance_id` dùng để làm gì và không phải loại ID nào?
- Đề xuất một quy tắc biên mask:
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
