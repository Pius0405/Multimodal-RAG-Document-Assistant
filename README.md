# 📄 Multimodal-RAG-Document-Assistant

## 🎯 Overview
This is a document-based chatbot that leverages Retrieval-Augmented Generation (RAG) to enable intelligent conversations with uploaded PDF documents. The system extracts both text and images from PDFs, processes them and provides responses based on user query. This project combines advanced natural language processing with computer vision to create a comprehensive document understanding system. Users can upload PDF files and ask questions about their content. The assistant retrieves relevant information from the documents and generates accurate responses grounded exclusively in the uploaded materials.

<img width="1600" height="689" alt="image" src="https://github.com/user-attachments/assets/86652241-5cd8-4a90-a425-98ff07b61177" />


## 🤖 Models Used
Language Model: **Qwen/Qwen2.5-7B-Instruct** - A 7 billion parameter instruction-tuned language model for generating contextual responses
Embedding Model: **all-MiniLM-L6-v2** - A lightweight sentence transformer for encoding documents and queries into semantic vectors
Vision Model: **nlpconnect/vit-gpt2-image-captioning** - A Vision Transformer + GPT-2 model for generating text descriptions of images extracted from PDFs

## 🏗️ Architecture
The system follows a modular architecture designed for scalability and maintainability

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)                 │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    ┌────▼──────┐          ┌─────▼──────┐
    │   Upload  │          │   Chat     │
    │  Handler  │          │  Interface │
    └────┬──────┘          └─────┬──────┘
         │                       │
    ┌────▼────────────────────────▼────┐
    │   PDF Processor (pdf_processor)  │
    │  - Text Extraction               │
    │  - Image Extraction & Captioning │
    └────┬─────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Knowledge Base Service           │
    │  (knowledge_base.py)              │
    │  - Text Chunking                  │
    │  - Embeddings Generation          │
    │  - MD5 Deduplication              │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   Vector Store (Chroma DB)        │
    │   (vector_stores.py)              │
    └───────────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   RAG Service (rag.py)            │
    │  - Document Retrieval             │
    │  - Context Formatting             │
    │  - Response Generation            │
    └───────────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Chat History Store               │
    │  (file_history_store.py)          │
    │  - Persistent Message Storage     │
    └───────────────────────────────────┘
```

## ✨ Features
1. 🖼️ Multimodal Document Processing
2. 💾 Long-Term Memory Storage: Conversation history is persisted to disk using JSON format
3. 🔄 File Caching & Deduplication :MD5 Hash Deduplication that prevents duplicate documents from being added to the vector store

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
