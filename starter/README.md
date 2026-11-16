# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `day6_retriever.py` | Dense query vào PostgreSQL/pgvector Day 6, strategy B |
| `gemini_controller.py` | Gemini API controller trả action search/answer dạng JSON |
| `run_agent.py` | Chạy agent thật và lưu trace JSON tùy chọn |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

Benchmark thật đã chuẩn bị ở [`data/benchmark_questions.json`](../data/benchmark_questions.json), gồm cả câu thiếu evidence. Baseline chỉ tính Hit@1, Hit@3 và Context Evidence Rate trên câu `answerable: true`; câu unanswerable vẫn được giữ trong trace nhưng không tính vào mẫu số retrieval. Chạy baseline với file chunks Day 6 đã xuất sang schema Day 7:

```bash
python starter/baseline.py --chunks <day6_strategy_b_chunks.json> --questions data/benchmark_questions.json --split dev --output starter/results/dev-tfidf
```

`data/agent_cases.json` là danh sách câu hỏi/nhãn dùng cho chấm agent; không truyền `benchmark_questions.json` hoặc gold vào Gemini/retriever.

Chạy agent thật cho một split, chỉ nạp `agent_cases.json` (không chứa đáp án vàng):

```bash
python starter/run_agent.py --cases data/agent_cases.json --split dev --output starter/results/agent-dev-traces.json
```

Sau khi chọn cấu hình trên dev, lặp lại với `--split test` và output path khác để giữ test held-out.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## 5. Kết nối Day 6 strategy B + Gemini

Adapter đã cấu hình strategy `B` và tái sử dụng `BAAI/bge-large-en-v1.5` (1024 chiều) để tạo query embedding. Model index đọc từ `DAY6_INDEX_EMBEDDING_MODEL`, tách biệt với `EMBEDDING_MODEL` dùng cho cấu hình ingest Day 6. Nó truy vấn bảng `ingestion_chunks` trong database Day 6 và chuyển `chunk_id`, `document_id`, nguồn, trang, metadata cùng score sang schema Day 7. Strategy B phải được ingest vào database trước; adapter không tự nạp lại hoặc tạo index.

1. Cài dependencies Day 6 trong cùng Python environment: `python -m pip install -r <đường-dẫn-Day6>/starter/requirements.txt`.
2. Khởi động PostgreSQL Day 6 (`docker compose -p day6_student_starter up -d --wait`) từ thư mục Day 6 `starter/`, rồi xác nhận `.env` ở đó có `DATABASE_URL` đúng.
3. Sao chép `.env.example` thành `.env` trong thư mục Day 7 `starter/`, điền `GEMINI_API_KEY` và chọn model bằng `GEMINI_MODEL`. Để đổi embedding query, chỉ sửa `DAY6_INDEX_EMBEDDING_MODEL` khi xác nhận index B đã được tạo bằng model đó. Không commit `.env`.
4. Chạy từ thư mục Day 7 `starter/`:

```bash
python run_agent.py "What does Git say about branching?" --top-k 3 --budget 3500 --output results/one-query.json
```

Mặc định adapter tìm repo Day 6 bên cạnh repo Day 7. Nếu đặt ở nơi khác, đặt `DAY6_STARTER` trỏ tới thư mục `starter/` của Day 6. Gemini chỉ điều phối và trả lời; không thay embedding model. Với model mặc định, adapter kiểm tra vector có 1024 chiều như index đã ghi nhận. `run_agent.py` lưu tool calls, retrieved chunks, context và citations để dùng làm trace; hãy tự review citation support theo benchmark.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
