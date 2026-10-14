# Báo cáo lỗi và lưu ý kỹ thuật

Tài liệu này ghi các bẫy cần kiểm tra trong quá trình chạy. Sau mỗi lỗi thực tế,
bổ sung triệu chứng, nguyên nhân, cách sửa và ảnh/log ngắn của lần chạy đó.

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
| Chưa chạy GPU | — | — | — | — | — |
