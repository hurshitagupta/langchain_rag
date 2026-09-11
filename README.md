# LangChain RAG Assessment

## Task 1 — Ingestion

### Objective

The goal of this task is to load, clean, and chunk a PDF containing at least 50 pages, while also recording measurable chunk statistics.

For this task, I used a PDF book containing **136 pages**.

### Implementation

The ingestion pipeline is implemented in:

```text
ingestion/ingestion.py
```

The pipeline performs the following steps:

1. Loads the PDF using `PyPDFLoader`.
2. Checks that the document contains at least 50 pages.
3. Cleans extracted text by removing unnecessary whitespace and line breaks.
4. Splits the pages into smaller chunks using `RecursiveCharacterTextSplitter`.
5. Adds a unique `chunk_id` to every generated chunk.
6. Preserves metadata such as source and page number.
7. Calculates chunk statistics including:

   * Total pages loaded
   * Total chunks created
   * Average chunk size
   * Smallest chunk size
   * Largest chunk size

### Chunking Configuration

```python
chunk_size = 900
chunk_overlap = 120
```

A chunk overlap is used so that important information near chunk boundaries is not completely separated between neighbouring chunks.

### Run the Ingestion Pipeline

```bash
uv run python -m ingestion.ingestion
```

### Save the Output

```bash
uv run python -m ingestion.ingestion > outputs/ingestion_output.txt
```

### Metadata

Each chunk retains document metadata and also receives a unique `chunk_id`.

This metadata will later be used during indexing and citation generation.

### Testing

Automated tests are included in:

```text
tests/test_ingestion.py
```

Run the tests using:

```bash
uv run pytest tests/test_ingestion.py -v
```

Save the test output using:

```bash
uv run pytest tests/test_ingestion.py -v > outputs/ingestion_test_output.txt
```

### Task 1 Result

Task 1 successfully demonstrates PDF ingestion, text cleaning, chunking, metadata preservation, validation of the minimum page requirement, and recorded chunk statistics.
