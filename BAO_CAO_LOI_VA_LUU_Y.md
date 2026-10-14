# Báo cáo lỗi và lưu ý kỹ thuật

## Mâu thuẫn giữa test template và checker cuối

`tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions`
yêu cầu `REPORT.md` còn chứa chuỗi `<điền>`, trong khi
`scripts/check_submission.py` đánh FAIL nếu chính chuỗi đó còn tồn tại. Hai điều
kiện không thể cùng đúng sau khi hoàn thiện báo cáo. Notebook đã chạy và lưu
`68 passed, 1 skipped` trước khi điền report như đề yêu cầu. Ở bước cuối, giữ
nguyên `scripts/` và `tests/`, chạy lại các test chức năng rồi lấy `0 FAIL` của
checker làm điều kiện nghiệm thu.

Tài liệu này ghi các bẫy kỹ thuật và các lỗi đã gặp trong lần chạy Kaggle cuối.
Các bản sửa tương ứng đã được đưa vào `lab_day10_solution.py`,
`finish_kaggle.py` và notebook.

## 1. Không trộn frame cùng session giữa các split

Chia ngẫu nhiên theo ảnh làm các frame gần như giống nhau rơi vào cả train và
test, khiến F1 đo khả năng nhớ bối cảnh thay vì tổng quát hóa. Luôn dùng split do
`scripts/prepare_data.py` sinh và gọi `dataset_summary()` để assert mỗi session
chỉ thuộc một split.

## 2. Nhầm thang toạ độ

Scorer mặc định hiểu `bbox_2d` ở thang 0–1000. Output pixel 640×360 dù JSON hợp
lệ vẫn cho IoU/F1 gần 0 nếu chấm như `norm1000`. Ablation pixel phải đổi đồng
thời prompt, hàm encode và thêm `--coord pixel` khi chấm.

## 3. Chat template không cùng tiền tố

Độ dài mask không nên ước lượng từ số token prompt thuần. Cần encode riêng
conversation user với `add_generation_prompt=True`, sau đó assert nó là tiền tố
của conversation đầy đủ. Nếu assertion lỗi, dừng thay vì train nhãn lệch.

## 4. Tính loss trên prompt/ảnh

Token ảnh và user prompt phải có label `-100`; padding của `labels` cũng phải là
`-100`. Chỉ token assistant được giữ nguyên ID. Nếu không, phần lớn gradient bị
dùng cho vùng không phải mục tiêu JSON.

## 5. Pad tensor thị giác

Chỉ pad `input_ids`, `attention_mask`, trường token-type (nếu có) và `labels`.
`pixel_values` cùng `image_grid_thw` phải `torch.cat` theo chiều 0. Pad hai tensor
này làm sai ánh xạ giữa image token và patch features.

## 6. Lượng tử hoá nhầm ViT/projector

Khi nạp NF4, cấu hình phải giữ `visual` và `lm_head` ngoài lượng tử hoá. Nếu
projector bị biến thành trọng số 4-bit, E2/E3 không thể cập nhật đầy đủ các tham
số merger như thiết kế.

## 7. Projector không ở float32

E2/E3 mở `visual.merger` và `visual.deepstack_merger_list`, đồng thời chuyển
chúng sang float32. Nếu loss thành NaN/Inf, kiểm tra dtype, mask, learning rate
và dừng ngay; không tiếp tục tạo evidence từ một run hỏng.

## 8. PEFT không tự lưu projector

`save_pretrained()` của PEFT lưu LoRA adapter nhưng không bảo đảm lưu base
parameter được mở thủ công. Vì thế E2/E3 phải có `projector.pt`. Khi suy luận,
phải nạp adapter rồi nạp đúng state projector; thiếu bước này có thể làm E2/E3
trông giống E1.

## 9. Prediction resume dùng nhầm model

`run_inference(..., resume=True)` bỏ qua mọi `image_id` đã có trong JSONL. Đây là
cơ chế chống mất phiên Colab, không phải cache dùng chung. Khi đổi checkpoint,
phải dùng file output khác hoặc đặt `resume=False` để tránh trộn prediction từ
hai model.

## 10. Số metadata không được chép từ đề

`trainable_params`, `peak_vram_gb` và `train_minutes` trong evidence phải đọc từ
`work/eN/run.json` của lần chạy thật. Các con số minh họa trong đề không phải kết
quả được phép nộp.

## 11. Đọc CI95 đúng cách

F1 điểm cao hơn chưa đủ để kết luận setup tốt hơn. Nếu CI95 chồng lấn đáng kể,
cần viết rằng test 311 ảnh chưa cung cấp đủ bằng chứng cho khác biệt chắc chắn.
Phân tích thêm F1 từng lớp, FP/ảnh và ví dụ định tính.

## 12. Reward hacking ở E5

Theo dõi tỷ lệ output `[]`, box toàn ảnh và số prediction/ảnh. Reward dùng
`max(N_gt, N_pred, 1)` cùng ngưỡng IoU 0.5 để phạt box dư/box khổng lồ, nhưng vẫn
phải so F1 test và xem output mẫu thay vì chỉ nhìn reward train tăng.

## Nhật ký lỗi thực tế

| Thời điểm | Setup/step | Triệu chứng | Nguyên nhân | Cách xử lý | Kết quả |
|---|---|---|---|---|---|
| 14/10/2026 | E2, bước optimizer đầu | `ValueError: Attempting to unscale FP16 gradients` tại `scaler.unscale_(optimizer)` | Một số tensor/gradient LoRA-projector kế thừa fp16 dù dùng GradScaler | Ép mọi tham số trainable và gradient tích luỹ sang float32 trước `unscale_`; thêm kiểm tra dtype trước train | E2/E3 train đủ 300 bước, loss hữu hạn |
| 14/10/2026 | Inference E2/E3 | Lỗi phép nhân do dtype projector và vision feature không khớp sau train | Projector được giữ float32 khi train nhưng backbone inference chạy fp16 | Lưu adapter + `projector.pt`, giải phóng model train rồi nạp lại artifact vào base model trước inference | E2/E3 suy luận đủ 311 ảnh và `missing_predictions=0` |
| 14–15/10/2026 | Kaggle đang chạy E3 | Kernel bị restart; module/repository dưới `/kaggle/working` không còn trong session mới | `/kaggle/working` là vùng tạm theo session; state Python và file chưa Save Version không bền vững | Tách notebook thành các cell theo E0–E4, mỗi chặng tạo ZIP backup; dùng `force=False` để bỏ qua evidence/artifact đã đủ | Lần chạy cuối khôi phục được theo chặng và xuất đủ gói nộp |
| 15/10/2026 | Tải ZIP kết quả | Link `FileLink` mở qua Jupyter proxy trả `404 page not found` | Route tải của draft session hết hiệu lực/không được trình duyệt giữ phiên | Tải file từ Kaggle Notebook Output/Files sau khi cell đóng gói hoàn tất | Đã lấy được `day10-submission` và đối chiếu checksum/nội dung trước khi chép vào repo |
| 15/10/2026 | Nghiệm thu local | `pytest -q` còn 1 lỗi vì test đòi `REPORT.md` chứa `<điền>` | Test kiểm tra template mâu thuẫn với checker cuối, vốn cấm placeholder | Giữ nguyên `tests/` và `scripts/`; chạy 67 test chức năng còn lại và dùng `check_submission.py` làm điều kiện nộp | `67 passed, 1 skipped, 1 deselected`; checker đạt `0 FAIL` |
