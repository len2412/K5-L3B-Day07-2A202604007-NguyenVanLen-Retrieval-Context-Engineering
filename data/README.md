# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

Benchmark Day 7 hiện nằm trong `benchmark_questions.json`: 15 câu có đáp án và 2 câu thiếu evidence ở dev; 5 câu có đáp án và 2 câu thiếu evidence ở test. Gold/evidence chỉ ở file benchmark để đánh giá offline. `agent_cases.json` chứa câu hỏi và nhãn answerable/abstain cho phần agent, không chứa đáp án chuẩn hoặc evidence.

Benchmark dùng cùng corpus Day 6: *Pro Git*, Scott Chacon và Ben Straub, bản 2.1.450; source ID `progit-v2-1-450`; 95 trang nội dung in số trang 10–104 thuộc chương 1–3. PDF gốc và manifest nằm trong project Day 6. Manifest ghi nguồn xuất bản và giấy phép CC BY-NC-SA 3.0; giữ attribution và điều kiện giấy phép nếu phân phối lại dữ liệu trích xuất. Từng evidence answerable đã được đối chiếu với trang nguồn và 177 chunk strategy B từ lần chạy Day 6.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).
