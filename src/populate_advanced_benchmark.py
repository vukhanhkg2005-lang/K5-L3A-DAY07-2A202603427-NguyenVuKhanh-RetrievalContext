"""Populate high quality factual benchmark questions for the Day 7 Advanced Challenge."""

import json
from pathlib import Path

dev_questions = [
    {
        "id": "ADV_DEV_01", "split": "dev", "category": "table",
        "question": "Trong Bảng 1 (Table 1, trang 5), mô hình GPT-4 đạt bao nhiêu điểm và xếp hạng percentile thứ mấy trên kỳ thi Uniform Bar Exam?",
        "expected_behavior": "answer",
        "gold": [{"page": 5, "chunk_id": "adv_p5_txt0", "evidence": "Uniform Bar Exam (MBE+MEE+MPT) 298 / 400 (~90th)"}]
    },
    {
        "id": "ADV_DEV_02", "split": "dev", "category": "table",
        "question": "Trong Bảng 2 (Table 2, trang 7), trên bộ benchmark lập trình HumanEval (0-shot Python coding tasks), kết quả độ chính xác của GPT-4 và GPT-3.5 lần lượt là bao nhiêu phần trăm?",
        "expected_behavior": "answer",
        "gold": [{"page": 7, "chunk_id": "adv_p7_txt1", "evidence": "HumanEval [43] 67.0% 48.1%"}]
    },
    {
        "id": "ADV_DEV_03", "split": "dev", "category": "table",
        "question": "Trong Bảng 8 (Table 8, trang 28), điểm số trung bình (average score) trên các bài thi giữa mô hình Base model và mô hình RLHF model lần lượt là bao nhiêu phần trăm?",
        "expected_behavior": "answer",
        "gold": [{"page": 28, "chunk_id": "adv_p28_tab0", "evidence": "base model achieves an average score of 73.7% while the RLHF model achieves an average score of 74.0%"}]
    },
    {
        "id": "ADV_DEV_04", "split": "dev", "category": "figure",
        "question": "Hình 1 (Figure 1, trang 3) minh họa cho quy luật dự đoán năng lực (scaling law) của GPT-4 dựa trên đại lượng đo lường nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 3, "chunk_id": "adv_p3_fig0", "evidence": "Figure 1. Performance of GPT-4 and smaller models. The metric is final loss"}]
    },
    {
        "id": "ADV_DEV_05", "split": "dev", "category": "figure",
        "question": "Hình 4 (Figure 4, trang 6) trực quan hóa kết quả đánh giá của GPT-4 trên những loại bài thi nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 6, "chunk_id": "adv_p6_fig0", "evidence": "Figure 4. GPT performance on academic and professional exams."}]
    },
    {
        "id": "ADV_DEV_06", "split": "dev", "category": "figure",
        "question": "Hình 5 (Figure 5, trang 8) so sánh độ chính xác của GPT-4 trên bộ benchmark MMLU 3-shot ở những ngôn ngữ nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 8, "chunk_id": "adv_p8_fig0", "evidence": "Figure 5. Performance of GPT-4 in a variety of languages compared to prior models in English on"}]
    },
    {
        "id": "ADV_DEV_07", "split": "dev", "category": "chart",
        "question": "Biểu đồ Hình 2 (Figure 2, trang 3) dự đoán năng lực của GPT-4 trên tập bài toán lập trình nào thuộc dataset nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 3, "chunk_id": "adv_p3_txt1", "evidence": "Capability prediction on 23 coding problems the HumanEval dataset."}]
    },
    {
        "id": "ADV_DEV_08", "split": "dev", "category": "chart",
        "question": "Biểu đồ Hình 3 (Figure 3, trang 4) đo lường độ chính xác của GPT-4 và các mô hình nhỏ hơn trên tác vụ nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 4, "chunk_id": "adv_p4_fig0", "evidence": "Figure 3. Performance of GPT-4 and smaller models on the Hindsight Neglect task."}]
    },
    {
        "id": "ADV_DEV_09", "split": "dev", "category": "chart",
        "question": "Biểu đồ Hình 7 (Figure 7, trang 11) đo lường hiệu năng của GPT-4 trên bộ benchmark TruthfulQA theo chiều hướng nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 11, "chunk_id": "adv_p11_fig0", "evidence": "Figure 7. Performance of GPT-4 on TruthfulQA. Accuracy is shown on the y-axis, higher is better."}]
    },
    {
        "id": "ADV_DEV_10", "split": "dev", "category": "multi_hop",
        "question": "Kết hợp thông tin từ trang 3 (Figure 1) và trang 4 (Figure 3), hai khía cạnh năng lực nào của mô hình được kiểm chứng về khả năng dự đoán được (predictable scaling) và tính chất khó dự đoán?",
        "expected_behavior": "answer",
        "gold": [{"page": 3, "chunk_id": "adv_p3_fig0", "evidence": "final loss"}, {"page": 4, "chunk_id": "adv_p4_fig0", "evidence": "Hindsight Neglect"}]
    },
    {
        "id": "ADV_DEV_11", "split": "dev", "category": "multi_hop",
        "question": "Kết hợp Bảng 1 (trang 5) và giải thích tại trang 6, phiên bản GPT-4 có xử lý hình ảnh (vision) và bản không có vision (no vision) trên kỳ thi Uniform Bar Exam đạt cùng mức điểm hay khác nhau?",
        "expected_behavior": "answer",
        "gold": [{"page": 5, "chunk_id": "adv_p5_txt0", "evidence": "Uniform Bar Exam (MBE+MEE+MPT) 298 / 400 (~90th)"}]
    },
    {
        "id": "ADV_DEV_12", "split": "dev", "category": "multi_hop",
        "question": "Đối chiếu kết quả Bảng 2 (trang 7) và Bảng 8 (trang 28), điểm số lập trình HumanEval và mức độ thay đổi điểm số trung bình sau giai đoạn RLHF cho thấy điều gì về năng lực cốt lõi của mô hình?",
        "expected_behavior": "answer",
        "gold": [{"page": 7, "chunk_id": "adv_p7_txt1", "evidence": "67.0%"}, {"page": 28, "chunk_id": "adv_p28_tab0", "evidence": "73.7% while the RLHF model achieves an average score of 74.0%"}]
    },
    {
        "id": "ADV_DEV_13", "split": "dev", "category": "text",
        "question": "Theo phần mở đầu trang 1, GPT-4 là mô hình thuộc loại gì và có khả năng xử lý những dạng dữ liệu đầu vào nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 1, "chunk_id": "adv_p1_txt0", "evidence": "large multimodal model"}]
    },
    {
        "id": "ADV_DEV_14", "split": "dev", "category": "text",
        "question": "Theo trang 2, một trong những mục tiêu kỹ thuật chính của dự án GPT-4 là phát triển hạ tầng tính toán (infrastructure) có đặc điểm gì?",
        "expected_behavior": "answer",
        "gold": [{"page": 2, "chunk_id": "adv_p2_txt0", "evidence": "predictable deep learning stack"}]
    },
    {
        "id": "ADV_DEV_15", "split": "dev", "category": "text",
        "question": "Theo phần kiểm tra an toàn tại trang 12, nhóm tác giả đã thu thập thêm dữ liệu nhằm cải thiện khả năng nào của GPT-4 khi gặp các yêu cầu độc hại?",
        "expected_behavior": "answer",
        "gold": [{"page": 12, "chunk_id": "adv_p12_txt2", "evidence": "improve GPT-4’s ability to refuse requests"}]
    },
    {
        "id": "ADV_DEV_16", "split": "dev", "category": "unanswerable",
        "question": "Chi phí tiền điện và tổng kinh phí tài chính cụ thể bằng đô la Mỹ (USD) để OpenAI xây dựng cụm siêu máy tính cho GPT-4 là bao nhiêu?",
        "expected_behavior": "abstain",
        "gold": []
    },
    {
        "id": "ADV_DEV_17", "split": "dev", "category": "unanswerable",
        "question": "Kiến trúc mô hình GPT-5 sử dụng bao nhiêu tham số và số lượng chuyên gia Mixture-of-Experts là bao nhiêu?",
        "expected_behavior": "abstain",
        "gold": []
    },
    {
        "id": "ADV_DEV_18", "split": "dev", "category": "unanswerable",
        "question": "Tỷ lệ phần trăm nhân viên của công ty Anthropic đồng ý với các kết quả an toàn của GPT-4 là bao nhiêu?",
        "expected_behavior": "abstain",
        "gold": []
    }
]

test_questions = [
    {
        "id": "ADV_TEST_01", "split": "test", "category": "table",
        "question": "Theo Bảng 1 (trang 5), trên kỳ thi Advanced Sommelier (kiến thức lý thuyết), tỷ lệ phần trăm điểm của GPT-4 là bao nhiêu so với 46% của GPT-3.5?",
        "expected_behavior": "answer",
        "gold": [{"page": 5, "chunk_id": "adv_p5_txt3", "evidence": "Advanced Sommelier (theory knowledge) 77 % 77 % 46 %"}]
    },
    {
        "id": "ADV_TEST_02", "split": "test", "category": "table",
        "question": "Theo Bảng 2 (trang 7), trên tác vụ lập trình HumanEval 0-shot, mô hình PaLM đạt bao nhiêu phần trăm so với 67.0% của GPT-4?",
        "expected_behavior": "answer",
        "gold": [{"page": 7, "chunk_id": "adv_p7_txt1", "evidence": "HumanEval [43] 67.0% 48.1% 26.2% 65.8% Python coding tasks 0-shot 0-shot 0-shot PaLM"}]
    },
    {
        "id": "ADV_TEST_03", "split": "test", "category": "table",
        "question": "Theo Bảng 5 (trang 12), nhóm tác giả đã làm việc với hơn bao nhiêu chuyên gia (experts) trong chiến dịch Expert Red Teaming?",
        "expected_behavior": "answer",
        "gold": [{"page": 12, "chunk_id": "adv_p12_tab0", "evidence": "over 50 experts from domains such as long-term AI alignment risks"}]
    },
    {
        "id": "ADV_TEST_04", "split": "test", "category": "table",
        "question": "Dựa vào Bảng 1 (trang 5), trên kỳ thi AP Microeconomics, điểm của GPT-4 được xếp vào thang điểm mấy và thuộc dải percentile nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 5, "chunk_id": "adv_p5_txt2", "evidence": "AP Microeconomics 5 (82nd - 100th)"}]
    },
    {
        "id": "ADV_TEST_05", "split": "test", "category": "table",
        "question": "Trong Bảng 8 (trang 28), trên bài thi LSAT (MCQ), điểm số của mô hình Base model là bao nhiêu?",
        "expected_behavior": "answer",
        "gold": [{"page": 28, "chunk_id": "adv_p28_txt1", "evidence": "Exam Base model RLHF model LSAT (MCQ) 67.0"}]
    },
    {
        "id": "ADV_TEST_06", "split": "test", "category": "table",
        "question": "Theo Bảng 1 (trang 5), trên bài thi Leetcode (easy), GPT-4 giải quyết được bao nhiêu câu trên tổng số câu?",
        "expected_behavior": "answer",
        "gold": [{"page": 5, "chunk_id": "adv_p5_txt3", "evidence": "Leetcode (easy) 31 / 41"}]
    },
    {
        "id": "ADV_TEST_07", "split": "test", "category": "figure",
        "question": "Theo Hình 1 (Figure 1, trang 3), đại lượng được biểu diễn trên trục tung (y-axis) để đánh giá năng lực là gì?",
        "expected_behavior": "answer",
        "gold": [{"page": 3, "chunk_id": "adv_p3_txt0", "evidence": "Bits per word"}]
    },
    {
        "id": "ADV_TEST_08", "split": "test", "category": "figure",
        "question": "Hình 5 (Figure 5, trang 8) đánh giá năng lực hiểu ngôn ngữ của GPT-4 trên bộ benchmark nào dưới thiết lập mấy-shot?",
        "expected_behavior": "answer",
        "gold": [{"page": 8, "chunk_id": "adv_p8_txt0", "evidence": "GPT-4 3-shot accuracy on MMLU across languages"}]
    },
    {
        "id": "ADV_TEST_09", "split": "test", "category": "figure",
        "question": "Hình 6 (Figure 6, trang 63) trong System Card mô tả cấu trúc mẫu câu nhắc (Example Prompt) cho thành phần an toàn nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 63, "chunk_id": "adv_p63_fig0", "evidence": "Figure 6: Example Prompt for RBRM"}]
    },
    {
        "id": "ADV_TEST_10", "split": "test", "category": "figure",
        "question": "Hình 10 (Figure 10, trang 68) trong phụ lục an toàn cung cấp các ví dụ minh họa về hiện tượng gì?",
        "expected_behavior": "answer",
        "gold": [{"page": 68, "chunk_id": "adv_p68_fig0", "evidence": "Figure 10: Example Jailbreaks for GPT-4-launch"}]
    },
    {
        "id": "ADV_TEST_11", "split": "test", "category": "figure",
        "question": "Hình 11 (Figure 11, trang 98) thể hiện kết quả đánh giá năng lực tuân thủ chỉ dẫn (IF evaluations) trên những mô hình nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 98, "chunk_id": "adv_p98_fig0", "evidence": "Figure 11: Results on IF evaluations across GPT3.5, GPT3.5-Turbo, GPT-4-launch"}]
    },
    {
        "id": "ADV_TEST_12", "split": "test", "category": "chart",
        "question": "Trên biểu đồ Hình 8 (Figure 8, trang 12), chỉ số sai số hiệu chuẩn kỳ vọng (ECE - Expected Calibration Error) của mô hình pre-trained GPT-4 trên tập con MMLU đạt giá trị là bao nhiêu?",
        "expected_behavior": "answer",
        "gold": [{"page": 12, "chunk_id": "adv_p12_txt0", "evidence": "ECE: 0.007 Calibration curve"}]
    },
    {
        "id": "ADV_TEST_13", "split": "test", "category": "chart",
        "question": "Trên biểu đồ Hình 9 (Figure 9, trang 14), tỷ lệ hành vi không đúng (Incorrect behavior rate) được đo lường trên hai loại câu nhắc nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 14, "chunk_id": "adv_p14_txt0", "evidence": "Sensitive Prompts Disallowed Prompts"}]
    },
    {
        "id": "ADV_TEST_14", "split": "test", "category": "chart",
        "question": "Biểu đồ Hình 3 (Figure 3, trang 4) so sánh độ chính xác của GPT-4 trên Hindsight Neglect với những mô hình tiền nhiệm nào của OpenAI?",
        "expected_behavior": "answer",
        "gold": [{"page": 4, "chunk_id": "adv_p4_txt2", "evidence": "ada, babbage, and curie"}]
    },
    {
        "id": "ADV_TEST_15", "split": "test", "category": "chart",
        "question": "Trên biểu đồ Hình 9 (Figure 9, trang 14), giá trị tỷ lệ hành vi không đúng (incorrect behavior rate) thấp hơn hay cao hơn thì được coi là an toàn/tốt hơn?",
        "expected_behavior": "answer",
        "gold": [{"page": 14, "chunk_id": "adv_p14_fig0", "evidence": "Lower values are better"}]
    },
    {
        "id": "ADV_TEST_16", "split": "test", "category": "chart",
        "question": "Biểu đồ Hình 6 (Figure 6, trang 10) thể hiện hiệu năng của GPT-4 trên bao nhiêu bài đánh giá tính trung thực nội bộ (factuality evaluations)?",
        "expected_behavior": "answer",
        "gold": [{"page": 10, "chunk_id": "adv_p10_fig0", "evidence": "Figure 6. Performance of GPT-4 on nine internal adversarially-designed factuality evaluations."}]
    },
    {
        "id": "ADV_TEST_17", "split": "test", "category": "multi_hop",
        "question": "So sánh kết quả giữa Bảng 1 (trang 5) và giải thích tại trang 6, trên kỳ thi Uniform Bar Exam, GPT-4 đạt khoảng percentile thứ bao nhiêu?",
        "expected_behavior": "answer",
        "gold": [{"page": 5, "chunk_id": "adv_p5_txt0", "evidence": "298 / 400 (~90th)"}, {"page": 6, "chunk_id": "adv_p6_txt0", "evidence": "Uniform Bar Exam"}]
    },
    {
        "id": "ADV_TEST_18", "split": "test", "category": "multi_hop",
        "question": "Kết hợp dữ liệu từ Bảng 2 (trang 7) và thảo luận tại trang 8 (Hình 5), ngôn ngữ tiếng Anh và các ngôn ngữ khác trên bài kiểm tra MMLU phản ánh điều gì về năng lực đa ngôn ngữ của GPT-4?",
        "expected_behavior": "answer",
        "gold": [{"page": 7, "chunk_id": "adv_p7_txt3", "evidence": "Latvian, Welsh, and Swahili"}, {"page": 8, "chunk_id": "adv_p8_txt0", "evidence": "GPT-4 3-shot accuracy on MMLU"}]
    },
    {
        "id": "ADV_TEST_19", "split": "test", "category": "multi_hop",
        "question": "Đối chiếu giữa Bảng 5 (trang 12) và Hình 9 (trang 14), việc hợp tác với hơn 50 chuyên gia Red Teaming đã trực tiếp giúp cải thiện chỉ số nào trong đánh giá an toàn?",
        "expected_behavior": "answer",
        "gold": [{"page": 12, "chunk_id": "adv_p12_tab0", "evidence": "over 50 experts"}, {"page": 14, "chunk_id": "adv_p14_fig0", "evidence": "Sensitive and disallowed prompts"}]
    },
    {
        "id": "ADV_TEST_20", "split": "test", "category": "multi_hop",
        "question": "So sánh giữa điểm thi lý thuyết của Base model và RLHF model tại Bảng 8 (trang 28) với việc hiệu chuẩn xác suất (calibration) tại trang 12 (Hình 8), quá trình post-training RLHF ảnh hưởng thế nào đến độ tin cậy xác suất của câu trả lời?",
        "expected_behavior": "answer",
        "gold": [{"page": 12, "chunk_id": "adv_p12_fig0", "evidence": "Calibration plot"}, {"page": 28, "chunk_id": "adv_p28_tab0", "evidence": "73.7% while the RLHF model achieves an average score of 74.0%"}]
    },
    {
        "id": "ADV_TEST_21", "split": "test", "category": "text",
        "question": "Theo tóm tắt tại trang 1, GPT-4 vượt trội hơn phần lớn các mô hình ngôn ngữ lớn hiện tại và thể hiện năng lực tương đương con người (human-level performance) trên lĩnh vực nào?",
        "expected_behavior": "answer",
        "gold": [{"page": 1, "chunk_id": "adv_p1_txt0", "evidence": "human-level performance on various professional and academic benchmarks"}]
    },
    {
        "id": "ADV_TEST_22", "split": "test", "category": "text",
        "question": "Theo thảo luận tại trang 6, nhóm tác giả đã áp dụng phương pháp nào để loại trừ nguy cơ dữ liệu bị rò rỉ (contamination check) giữa tập huấn luyện và các đề thi chuẩn hóa?",
        "expected_behavior": "answer",
        "gold": [{"page": 6, "chunk_id": "adv_p6_txt3", "evidence": "portions of BIG-bench [48] were inadvertently mixed into the training"}]
    },
    {
        "id": "ADV_TEST_23", "split": "test", "category": "text",
        "question": "Theo thảo luận tại trang 11, kỹ thuật Model-assisted Safety Pipeline và phương pháp RBRM được áp dụng trong giai đoạn nào của quá trình huấn luyện?",
        "expected_behavior": "answer",
        "gold": [{"page": 11, "chunk_id": "adv_p11_txt2", "evidence": "model-assisted safety pipeline [69]"}]
    },
    {
        "id": "ADV_TEST_24", "split": "test", "category": "text",
        "question": "Theo trang 14, ngoài việc từ chối các câu nhắc độc hại, hệ thống an toàn của GPT-4 còn được tối ưu để tránh hiện tượng gì đối với các câu nhắc an toàn thông thường?",
        "expected_behavior": "answer",
        "gold": [{"page": 14, "chunk_id": "adv_p14_txt1", "evidence": "iterative model improvement"}]
    },
    {
        "id": "ADV_TEST_25", "split": "test", "category": "unanswerable",
        "question": "Mức lương thưởng trung bình bằng tiền mặt USD của các kỹ sư trong danh sách đóng góp ở cuối bài báo là bao nhiêu?",
        "expected_behavior": "abstain",
        "gold": []
    },
    {
        "id": "ADV_TEST_26", "split": "test", "category": "unanswerable",
        "question": "Tọa độ địa lý GPS chính xác của trung tâm dữ liệu đặt cụm máy chủ huấn luyện GPT-4 nằm ở bang nào?",
        "expected_behavior": "abstain",
        "gold": []
    },
    {
        "id": "ADV_TEST_27", "split": "test", "category": "unanswerable",
        "question": "Doanh thu thương mại cụ thể bằng USD mà OpenAI thu được từ dịch vụ ChatGPT Plus trong năm 2023 được công bố ở bảng nào?",
        "expected_behavior": "abstain",
        "gold": []
    },
    {
        "id": "ADV_TEST_28", "split": "test", "category": "unanswerable",
        "question": "Biểu đồ nào trong bài báo so sánh trực tiếp hiệu năng của GPT-4 với mô hình Llama 3 405B của Meta?",
        "expected_behavior": "abstain",
        "gold": []
    },
    {
        "id": "ADV_TEST_29", "split": "test", "category": "unanswerable",
        "question": "Tên hãng xe ô tô điện mà CEO OpenAI đã lái tới dự hội nghị công bố GPT-4 được nhắc tới ở trang bao nhiêu?",
        "expected_behavior": "abstain",
        "gold": []
    },
    {
        "id": "ADV_TEST_30", "split": "test", "category": "unanswerable",
        "question": "Số lượng chip phần cứng TPU v5e mà Google đã bán cho OpenAI để huấn luyện GPT-4 là bao nhiêu?",
        "expected_behavior": "abstain",
        "gold": []
    }
]

def main():
    all_bench = dev_questions + test_questions
    out_file = Path("data/advanced_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_bench, f, ensure_ascii=False, indent=2)

    dev_c = [c for c in all_bench if c["split"] == "dev"]
    test_c = [c for c in all_bench if c["split"] == "test"]
    print(f"[+] Da cap nhat {len(all_bench)} cau hoi benchmark chat luong cao vao {out_file}")
    print(f"    - Dev: {len(dev_c)} cau (3 table, 3 fig, 3 chart, 3 multi-hop, 3 text, 3 unanswerable)")
    print(f"    - Test: {len(test_c)} cau (6 table, 5 fig, 5 chart, 4 multi-hop, 4 text, 6 unanswerable)")

if __name__ == "__main__":
    main()

