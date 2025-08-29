# RAG for PDF Files

A simple local demo application that implements **Retrieval-Augmented Generation (RAG)** for PDF files. This tool allows users to ask natural-language questions about the content of PDF documents, retrieving relevant text chunks and generating concise answers using a large language model (LLM).

---

## About

This project provides a minimal, extensible demo for applying Retrieval-Augmented Generation (RAG) to **PDF files**. Users can query the content of PDF documents using natural language, and the system retrieves relevant text chunks before generating answers with an LLM. 

This prototype is designed for local experimentation and serves as a foundation for integrating advanced vector stores or production-grade LLM APIs.

---

## Features

- Supports loading PDF files (`.pdf`) from the `input_files/` directory.
- Extracts text from PDFs and converts it into textual chunks for retrieval.
- Generates embeddings for chunks using an embedding provider or a local fallback.
- Implements an in-memory vector store with optional disk persistence.
- Provides a query interface via a command-line interface (CLI) or a web server.
- Includes a demo application (`app.py`) for script-based or server-based workflows.
- Configurable via environment variables for flexibility.

---

## Repository Structure

```plaintext
RAG-For-PDF-Files/
├── input_files/          # Directory for PDF files
├── app.py                # Main application/demo entry point
├── requirements.txt      # Python dependencies
├── LICENSE               # GPL-3.0 license file
└── README.md             # This file

```
# Requirements

- Python: 3.10 or higher
- pip: Python package manager
- Dependencies (listed in requirements.txt):

- PyPDF2 or pdfplumber (for PDF text extraction)
- numpy
- scikit-learn
- sentence-transformers (or another embedding provider)
- flask (required for web demo)
- openai (optional, for LLM API integration)

Installation (Local)

Clone the repo:
```
git clone https://github.com/BrijeshRakhasiya/RAG-For-Tabluer-Files-.git
cd RAG-For-Tabluer-Files-
```

Create and activate a virtual environment:
```
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS / Linux
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Add your tabular files into input_files/.

Configure environment variables (if using an external LLM / embedding API):
```
export OPENAI_API_KEY="sk-..."
```

Quick Start
Run in Script Mode : 
```
python app.py --mode script --file input_files/your_table.csv
```



# License

- This project is licensed under the GNU General Public License v3.0 — see the LICENSE
 file.

# Contact

- Author: **Brijesh Rakhasiya**
