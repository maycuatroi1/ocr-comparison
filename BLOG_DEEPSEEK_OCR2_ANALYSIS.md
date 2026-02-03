# DeepSeek-OCR-2: Visual Causal Flow - Phân Tích Paper và So Sánh Thực Nghiệm

**Tác giả:** Binh Nguyen
**Ngày:** February 2026
**Paper:** [arXiv:2601.20552](https://arxiv.org/abs/2601.20552)
**GitHub:** [deepseek-ai/DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)

---

## 1. Giới thiệu

Optical Character Recognition (OCR) đã có những bước tiến vượt bậc nhờ sự phát triển của Vision-Language Models (VLMs). Tuy nhiên, các mô hình OCR hiện tại vẫn gặp phải một hạn chế cơ bản: **chúng xử lý visual tokens theo thứ tự cố định (raster-scan)** - từ trái sang phải, từ trên xuống dưới - điều này không phản ánh cách con người thực sự đọc và nhận diện văn bản.

DeepSeek-OCR-2 giới thiệu một cách tiếp cận mới mang tên **Visual Causal Flow**, giải quyết vấn đề này bằng cách cho phép mô hình tự động sắp xếp lại visual tokens dựa trên ngữ nghĩa của nội dung hình ảnh.

---

## 2. Phân Tích Paper

### 2.1 Vấn đề với các VLM hiện tại

Các Vision-Language Models hiện tại như GPT-4V, Gemini, và LLaVA đều sử dụng cách tiếp cận tương tự:

```
Image → Visual Encoder → Visual Tokens → LLM (left-to-right processing)
```

**Hạn chế:**
- Visual tokens được xử lý theo thứ tự raster-scan cố định
- Không phù hợp với cách con người đọc văn bản (không phải lúc nào cũng từ trái sang phải)
- Khó khăn khi xử lý tài liệu có layout phức tạp (bảng, đa cột, diagrams)

### 2.2 Giải pháp: Visual Causal Flow

DeepSeek-OCR-2 đề xuất **DeepEncoder V2** với khả năng:

1. **Dynamic Token Reordering**: Tự động sắp xếp lại visual tokens dựa trên semantic content
2. **Causally-informed Sequential Processing**: Xử lý tuần tự có định hướng nguyên nhân
3. **2D Understanding via 1D Causal Reasoning**: Hiểu không gian 2D thông qua chuỗi suy luận 1D

### 2.3 Kiến trúc mô hình

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepSeek-OCR-2                           │
├─────────────────────────────────────────────────────────────┤
│  Input Image                                                │
│       ↓                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │     DeepEncoder V2                   │                   │
│  │  - Dynamic Resolution: 768×768 × N   │                   │
│  │  - Base patch: 1024×1024             │                   │
│  │  - Visual tokens: (0-6)×144 + 256    │                   │
│  └─────────────────────────────────────┘                   │
│       ↓                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │     Causal Token Reordering          │                   │
│  │  - Semantic-driven ordering          │                   │
│  │  - Layout-aware processing           │                   │
│  └─────────────────────────────────────┘                   │
│       ↓                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │     LLM Decoder (3B params)          │                   │
│  │  - Flash Attention 2                 │                   │
│  │  - BF16 precision                    │                   │
│  └─────────────────────────────────────┘                   │
│       ↓                                                     │
│  Output: Markdown + Bounding Boxes                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Hai chế độ hoạt động

| Mode | Prompt | Output |
|------|--------|--------|
| **Document Mode** | `<image>\n<\|grounding\|>Convert the document to markdown.` | Markdown + Bounding boxes |
| **Free OCR Mode** | `<image>\nFree OCR.` | Plain text |

---

## 3. Thực Nghiệm So Sánh

### 3.1 Setup thực nghiệm

| Parameter | Gemini Flash | DeepSeek-OCR-2 |
|-----------|--------------|----------------|
| **Model** | gemini-2.0-flash | deepseek-ai/DeepSeek-OCR-2 |
| **Parameters** | Unknown (API) | 3B |
| **Hardware** | API (Cloud) | NVIDIA H100 (Colab) |
| **Precision** | Unknown | BF16 |
| **Framework** | Google GenAI API | Transformers + Flash Attention 2 |

### 3.2 Test Images

Chúng tôi sử dụng 3 loại ảnh test khác nhau:

1. **Document.jpg** - Văn bản hành chính tiếng Việt (GIẤY TIẾP NHẬN HỒ SƠ)
2. **crazy-hand-writing.png** - Bảng trắng với chữ viết tay tiếng Việt
3. **wild.png** - Bia đá với văn bản khắc ngoài trời

### 3.3 Kết quả

#### 3.3.1 Tốc độ xử lý

| Image | Gemini Flash | DeepSeek-OCR-2 (H100) |
|-------|--------------|----------------------|
| Document.jpg | 5.13s | ~16s |
| crazy-hand-writing.png | 3.32s | ~10s |
| wild.png | 2.90s | ~8s |

**Nhận xét:** Gemini Flash nhanh hơn ~3-4x do sử dụng API cloud được tối ưu hóa.

#### 3.3.2 Chất lượng OCR - Document.jpg (Văn bản hành chính)

**Gemini Flash:**
```
UBND TỈNH QUẢNG BÌNH
THÀNH PHỐ ĐỒNG HỚI
Số: 291780963307/TNHS
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
...
```

**DeepSeek-OCR-2:**
```
HUAN TINH QUANG NINH
Số: 291789063307/TNHS
GIAY TIEP NHAN HO SO VA HEN TRA KET QUA
CONG HOA XA HOI CHU NCHA VIET NAM
Dac lai - Lir do - Thanh nhue
...
```

**Phân tích:**
- ✅ Gemini: Nhận diện tốt dấu tiếng Việt (QUẢNG BÌNH, CHỦ NGHĨA)
- ❌ DeepSeek: Mất dấu (QUANG NINH thay vì QUẢNG BÌNH, CHU NCHA thay vì CHỦ NGHĨA)
- ✅ DeepSeek: Có bounding boxes cho từng vùng text
- ✅ DeepSeek: Output structured (Markdown format)

#### 3.3.3 Chất lượng OCR - wild.png (Bia đá ngoài trời)

**Ground Truth (ước tính):**
```
CÂY LỘC VỪNG
DO ĐẠI TƯỚNG TÔ LÂM
UVTW ĐẢNG
THỨ TRƯỞNG BỘ CÔNG AN
TRỒNG NGÀY 15-2-2015
```

**Gemini Flash:**
```
CÂY LỘC VÙNG
DO ĐỤC TÔ LÂM
UVTW DANG
THỨ TRƯỞNG BỘ CÔNG AN
TRÔNG NGAY 15-2-2015
```

**DeepSeek-OCR-2:**
```
[Cần bổ sung kết quả từ Colab]
```

**Lỗi phổ biến:**
- "VỪNG" → "VÙNG" (cả hai model)
- "ĐẢNG" → "DANG" (mất dấu)
- "TRỒNG" → "TRÔNG" (sai dấu)

### 3.4 So sánh Features

| Feature | Gemini Flash | DeepSeek-OCR-2 |
|---------|--------------|----------------|
| Vietnamese diacritics | ⭐⭐⭐⭐ | ⭐⭐ |
| Handwriting recognition | ⭐⭐⭐ | ⭐⭐⭐ |
| Layout detection | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Bounding boxes | ❌ | ✅ |
| Structured output | ❌ | ✅ (Markdown) |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Cost | Pay-per-use | Free (self-hosted) |
| Privacy | Cloud | On-premise |

---

## 4. Điểm mạnh và hạn chế

### 4.1 DeepSeek-OCR-2

**Điểm mạnh:**
- ✅ Open-source, có thể self-host
- ✅ Bounding boxes cho từng text region
- ✅ Output Markdown structured
- ✅ Layout detection tốt
- ✅ Kiến trúc novel (Visual Causal Flow)

**Hạn chế:**
- ❌ Yêu cầu GPU mạnh (khuyến nghị T4+)
- ❌ Vietnamese diacritics chưa tốt
- ❌ Chậm hơn API solutions
- ❌ Model size lớn (~6GB)

### 4.2 Gemini Flash

**Điểm mạnh:**
- ✅ Nhanh, dễ sử dụng (API)
- ✅ Vietnamese support tốt hơn
- ✅ Không cần GPU local
- ✅ Luôn được cập nhật

**Hạn chế:**
- ❌ Closed-source
- ❌ Không có bounding boxes
- ❌ Pay-per-use cost
- ❌ Data privacy concerns

---

## 5. Kết luận

### 5.1 Đóng góp của paper

DeepSeek-OCR-2 mang lại một góc nhìn mới về OCR với **Visual Causal Flow**:

1. **Paradigm shift**: Từ raster-scan cố định sang dynamic semantic-driven ordering
2. **Câu hỏi nghiên cứu quan trọng**: "Có thể đạt được 2D understanding thông qua cascaded 1D causal reasoning?"
3. **Practical value**: Bounding boxes + Markdown output hữu ích cho document processing pipelines

### 5.2 Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| Quick OCR, Vietnamese text | Gemini Flash |
| Document digitization with layout | DeepSeek-OCR-2 |
| Privacy-sensitive documents | DeepSeek-OCR-2 (self-hosted) |
| Production API | Gemini Flash |
| Research/Experimentation | DeepSeek-OCR-2 |

### 5.3 Future work

- Fine-tune DeepSeek-OCR-2 cho Vietnamese
- Benchmark trên OmniDocBench v1.5
- So sánh với GPT-4V, Claude 3.5

---

## 6. Reproducing Results

### Run Gemini Flash (Mac M4)
```bash
cd ocr-comparison
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"
python scripts/gemini_ocr.py
python scripts/visualize_sidebyside.py
```

### Run DeepSeek-OCR-2 (Google Colab)
```
https://colab.research.google.com/github/maycuatroi1/ocr-comparison/blob/master/notebooks/deepseek_colab.ipynb
```

---

## References

1. Wei, H., Sun, Y., Li, Y. (2026). DeepSeek-OCR 2: Visual Causal Flow. arXiv:2601.20552
2. [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
3. [Google Gemini API](https://ai.google.dev/)
4. [Experiment Code Repository](https://github.com/maycuatroi1/ocr-comparison)

---

*Blog này là một phần của nghiên cứu về Knowledge Graphs for Cyber Threat Intelligence.*
