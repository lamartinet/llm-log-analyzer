# LLM Log Analyzer

Generate vector embeddings from log files for similarity search and analysis.

## How it works

1. Reads a log file line by line, skipping empty lines
2. Generates a 384-dimensional embedding for each line using `paraphrase-multilingual-MiniLM-L12-v2` (supports 50+ languages including Portuguese)
3. Saves embeddings as a `.npy` file via memory-mapped I/O (low RAM usage, suitable for large files)

## Install

```bash
pip install -e .
```

## Usage

```bash
python main.py
```

Or from Python:

```python
from main import read_log_file, generate_embeddings

lines = read_log_file("data/server.log")
count, dim = generate_embeddings(lines, "data/server.embeddings.npy")
print(f"Saved {count} embeddings of dimension {dim}")
```

## Output

Embeddings are saved as a float32 numpy array with shape `(N, 384)`:

```python
import numpy as np

embeddings = np.load("data/server.embeddings.npy", mmap_mode="r")
print(embeddings.shape)  # (100000, 384)
print(embeddings[0][:5])  # first 5 dimensions of line 0
```

## Model

[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) via FastEmbed (ONNX runtime).

| Property | Value |
|----------|-------|
| Dimensions | 384 |
| Size | 0.22 GB |
| Languages | 50+ |

## Dependencies

- Python >= 3.10
- fastembed
- numpy
- tqdm
