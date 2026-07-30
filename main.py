import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _detect_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using {device.upper()}")
    return device

def read_log_file(file_path: str) -> list[str]:
    """Read a log file and return non-empty lines with trailing newlines stripped."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.rstrip("\n")]


def generate_embeddings(
    lines: list[str],
    output_file: str,
    batch_size: int = 500,
) -> tuple[int, int]:
    """
    Generate embeddings for each line and save to a .npy file using memmap.

    Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim,
    multilingual, fast). Writes directly to disk via numpy memmap — only the
    current batch is kept in RAM, so memory usage stays low even for large files.

    Args:
        lines: Non-empty log lines to embed.
        output_file: Path to the .npy output file.
        batch_size: Number of lines processed per batch.

    Returns:
        A tuple of (number_of_embeddings, embedding_dimension).
    """
    device = _detect_device()
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=device,
    )

    for i in tqdm(range(0, len(lines), batch_size), desc="Generating embeddings"):
        batch = lines[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if i == 0:
            dim = batch_embeddings.shape[1]
            mmap = np.memmap(output_file, dtype="float32", mode="w+", shape=(len(lines), dim))

        mmap[i:i + len(batch)] = batch_embeddings

    mmap.flush()
    return len(lines), dim


if __name__ == "__main__":
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    input_file = "data/server.log.2026-07-16"
    output_file = "data/server.log.2026-07-16.embeddings.npy"

    lines = read_log_file(input_file)
    print("\n ********************* File Read Complete *********************")
    print(f"Read {len(lines)} lines.")

    count, dim = generate_embeddings(lines, output_file)
    print("\n ********************* Embeddings Generation Complete *********************")
    print(f"Saved {count} embeddings of dimension {dim} to {output_file}")
