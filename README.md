# Greek-Nyaya-RAG-System
A comparative Semantic Retrieval System for Greek Logic and Nyaya/Tarka Sastra using RAG
# A Comparative Semantic Retrieval System for Greek Logic and Nyaya/Tarka Sastra

A domain-specific, bidirectional Retrieval-Augmented Generation (RAG) system connecting classical Greek Logic and Indian Nyaya/Tarka Sastra using dense vector search and local LLM generation.

**Developer:** Taticherla Hasini (IIITDM Kurnool)  

---

## Features
* **Corpus Ingestion & Preprocessing:** Extracts, cleans, and normalizes raw text from foundational logic PDFs using `PyPDF` and regular expressions.
* **Context-Aware Chunking:** Paragraph-based sliding-window segmentation (800–1200 characters) preserving semantic context and argument premises.
* **Dense Vector Indexing:** Generates 1024-dimensional embeddings using `multilingual-e5-large-instruct` and indexes them into a local `Qdrant` vector database.
* **Source-Grounded Retrieval:** Uses Cosine Similarity to retrieve top-$k$ relevant context passages alongside source file and chunk metadata.
* **Grounded Answer Generation:** Formats RAG prompts and passes context into `Qwen2.5-3B-Instruct` to generate hallucination-free, cited responses.

---

## Project Structure

```text
.
├── scripts/
│   ├── 01_extract_clean.py        # PDF text extraction & normalization
│   ├── 02_chunk_metadata.py       # Paragraph chunking & metadata JSON creation
│   ├── 03_embeddings.py           # Embedding generation & Qdrant vector indexing
│   ├── 04_test_retrieval.py       # Testing the retrieval of Qdrant embeddding
│   ├── 05_retriever.py            # Vector search & top-k chunk retrieval module
│   └── 06_generator.py            # Prompt formatting & Qwen2.5 LLM generation
├── corpus/                        # Local corpus directory for reference PDFs
├── .gitignore                     # Excludes heavy datasets, cache, and virtual envs
├── requirements.txt               # Project dependencies
├── workflow_pipeline.ipynb        # Complete execution notebook (Google Colab)
├── workflow_project_report.pdf    # Comprehensive research & project report
└── README.md                      # Project documentation
```

---

## Installation

### Prerequisites

* Python 3.10 or higher
* CUDA-compatible GPU (recommended for local LLM execution)

### Setup Steps

1. Clone the repository and enter the directory:
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/Greek-Nyaya-RAG-System.git
   cd Greek-Nyaya-RAG-System
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Setup & Prerequisites

Before running the data processing scripts, place your source PDFs inside the `corpus/` directory as in the corpus.txt.


## Usage

### 1. Execute Data Preparation & Vector Indexing

Run the sequential data pipeline scripts to build the local Qdrant vector database:

```bash
# Step 1: Extract and clean raw PDF text
python scripts/01_extract_clean.py

# Step 2: Generate paragraph chunks and metadata
python scripts/02_chunk_metadata.py

# Step 3: Embed text chunks and store vectors in Qdrant
python scripts/03_embeddings.py
```

### 2. Run the RAG Query Pipeline

Launch the interactive RAG interface to query the corpus:

```bash
python scripts/generator.py
```

---

## Example Execution Output

```text
============================================
        RAG PIPELINE INITIATED     
============================================

Device: cuda
GPU: NVIDIA A100-SXM4-40GB

Loading embedding model...
Embedding model loaded!

Loading LLM...
LLM loaded!

Connected to Qdrant!

Enter your question: what is nyaya?

Retrieved 6 chunks from Qdrant.

Generating answer...

================================================================================
FINAL RAG ANSWER
================================================================================
Nyaya is one of the six orthodox systems of Hindu philosophy, primarily concerned
with logic, epistemology, and valid reasoning. It establishes rules for obtaining
valid knowledge (Prama) through four distinct sources (Pramanas): Perception (Pratyaksha),
Inference (Anumana), Comparison (Upamana), and Testimony (Sabda).

================================================================================

SOURCES USED:
--------------------------------------------------------------------------------
1. cp/ca1.txt (Chunk 3) [Score: 0.8516]
2. nyaya/Nyaya-Sutras-vol-1.txt (Chunk 1305) [Score: 0.8278]
3. nyaya/Nyaya-Sutras-vol-1.txt (Chunk 1298) [Score: 0.8235]
4. nyaya/Nyaya-Sutras-vol-1.txt (Chunk 14) [Score: 0.8222]
5. nyaya/Nyaya-Sutras-vol-1.txt (Chunk 377) [Score: 0.8221]
6. nyaya/Nyaya-Sutras-vol-1.txt (Chunk 21) [Score: 0.8207]
```

---

## Tech Stack & Models

* **Language:** Python 3.10+
* **PDF Processing:** PyPDF
* **Embedding Model:** `intfloat/multilingual-e5-large-instruct` (1024-dim)
* **Vector Database:** Qdrant (Cosine Distance Metric)
* **LLM Engine:** `Qwen/Qwen2.5-3B-Instruct`
* **Core Libraries:** PyTorch, Transformers, Sentence-Transformers, Accelerate

---

## Contributors

* **Taticherla Hasini** (Developer) - IIITDM Kurnool

---

## License

Distributed under the MIT License.
