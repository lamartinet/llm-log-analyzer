import numpy as np
from fastembed import TextEmbedding
from tqdm import tqdm


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
    model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    for i in tqdm(range(0, len(lines), batch_size), desc="Generating embeddings"):
        batch = lines[i:i + batch_size]
        batch_embeddings = list(model.embed(batch))

        if i == 0:
            # First batch: discover embedding dimension and allocate the memmap file
            dim = batch_embeddings[0].shape[0]
            mmap = np.memmap(output_file, dtype="float32", mode="w+", shape=(len(lines), dim))

        # Write batch directly to disk
        for j, emb in enumerate(batch_embeddings):
            mmap[i + j] = emb

    mmap.flush()
    return len(lines), dim


if __name__ == "__main__":
    input_file = "data/server.log.2026-07-16"
    output_file = "data/server.log.2026-07-16.embeddings.npy"

    lines = read_log_file(input_file)
    print("\n ********************* File Read Complete *********************")
    print(f"Read {len(lines)} lines.")

    count, dim = generate_embeddings(lines, output_file)
    print("\n ********************* Embeddings Generation Complete *********************")
    print(f"Saved {count} embeddings of dimension {dim} to {output_file}")