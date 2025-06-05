# NewsBot: AI-Powered News Q&A

NewsBot is your go-to AI assistant for quickly extracting insights from news articles and documents. Whether you provide a URL or upload a PDF, NewsBot intelligently processes the content, comprehends its context, and answers your questions using advanced language models. Powered by **Streamlit**, **LangChain**, and **OpenAI**, it seamlessly integrates real-time document parsing, **FAISS** for semantic search, and natural language understanding to deliver fast, accurate, and source-backed answers.

## Features

1. Accepts **URLs** or **PDF uploads** as content sources  <br>
2. Splits and embeds text using **OpenAI Embeddings** <br>
3. Fast and accurate Q&A using **FAISS vector search** <br>
4. Displays answers along with the **original source**  <br>
5. Powered by **LangChain** and **OpenAI GPT models** <br>

## NewsBot Architecture 

```
                        ┌─────────────────────────────┐
                        │        Streamlit UI         │
                        │ - Sidebar: URL/PDF inputs   │
                        │ - Main: Question + Answer   │
                        └────────────┬────────────────┘
                                     │
                            ▼ Process Trigger
                      (when user clicks "Load Source")
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────┐
        │          Content Loading & Preprocessing            │
        │─────────────────────────────────────────────────────│
        │ If URL(s):                                          │
        │   → UnstructuredURLLoader fetches & parses text     │
        │                                                     │
        │ If PDF:                                             │
        │   → PyPDF2 reads pages                              │
        │   → LangChain `Document` created from text          │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────────┐
        │       Recursive Text Chunking (LangChain)           │
        │ - Uses RecursiveCharacterTextSplitter               │
        │ - Breaks content into 1000-token overlapping chunks │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────────┐
        │         OpenAI Embeddings + FAISS Indexing          │
        │ - Embeds chunks using OpenAIEmbeddings              │
        │ - Stores vectors in FAISS index                     │
        │ - Saves (index, docstore, id_map) in pickle file    │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────────┐
        │          Question-Answer Inference Pipeline         │
        │ - User asks question                                │
        │ - FAISS index loaded                                │
        │ - Vector similarity search retrieves top docs       │
        │ - LangChain `RetrievalQAWithSourcesChain` runs LLM  │
        │ - GPT generates final answer using retrieved docs   │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
                        ┌───────────────┐
                        │  UI Output    │
                        │ - Answer      │
                        │ - Source(s)   │
                        └───────────────┘

```

## 📸 Screenshot

![Index](static/Index.png)


## 🔮 Future Enhancements

1. Image-Based Text Extraction
- Integrating Optical Character Recognition (OCR) to extract and analyze text from images embedded within documents or uploaded directly.

2. Handwritten & Scanned Document Support
- Extending compatibility to scanned PDFs and handwritten content using OCR tools, enabling processing of a broader range of document types.

3. Multi-Document Cross Analysis
- Allowing users to submit multiple documents or articles simultaneously for comparison, aggregation, and context-aware question answering.


## Author

👤 **[Amritangshu Dey](https://github.com/amritofficial88)**

## Connect With Me 🌐

**[![LinkedIn](https://img.shields.io/badge/LinkedIn-Amritangshu%20Dey-bluen)](https://www.linkedin.com/in/amritangshu-dey-400940251/)**


<p align="center"><b>© Created by Amritangshu Dey</b></p?
