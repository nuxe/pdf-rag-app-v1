# PDF RAG App

Vision-based retrieval-augmented generation for PDF documents using ColPali and Qwen2-VL.

## Technical Stack

- **ColQwen2**: Multi-vector embeddings for PDF page retrieval
- **Qwen2-VL-2B**: Vision-language model for text generation
- **Modal**: GPU compute with A100-80GB instances
- **pdf2image**: PDF to JPEG conversion via poppler

## Implementation

Uses two-stage RAG pipeline:
1. **Retrieval**: ColPali embeddings match query to most relevant PDF page
2. **Generation**: Qwen2-VL processes retrieved image + conversation history

Session state persisted in Modal Dict with PDF images stored on Modal Volume.

## Usage

```bash
modal run chat_with_pdf_vision.py --pdf-path document.pdf --question "query"
```