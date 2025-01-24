# RAG-Bot (Retrieval-Augmented Generation Bot)

## Overview  
A **Retrieval-Augmented Generation (RAG) bot** that processes documents, retrieves relevant information using **FAISS**, and generates responses with a **language model**. Designed for efficient and context-aware question answering.  

## Project Structure  
- `create_faiss_index.py` – Creates the FAISS index from the documents.  
- `rag_bot.py` – Builds the bot and processes queries using the index and language model.  

## Requirements  
Ensure you have the following dependencies installed:  
- `faiss`
- `transformers`
- `sentence-transformers`
- `torch`
- `llama_index`  

Install all dependencies with:  
```bash
pip install -r requirements.txt
```

## Usage  
### 1. Create the FAISS index  
Run the following command to process the documents and create the FAISS index:  
```bash
python create_faiss_index.py
```

### 2. Start the RAG bot  
Once the index is created, start the bot to handle queries:  
```bash
python rag_bot.py
```

### 3. Ask a question  
After running `rag_bot.py`, you can input queries, and the bot will return context-aware answers based on the indexed documents.  

## License & Usage  
⚠️ **This project is protected and for personal use only. Commercial or public use is not allowed.**  

