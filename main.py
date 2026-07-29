import os

import numpy as np
from fastembed import TextEmbedding
from tqdm import tqdm


def read_log_file(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.rstrip("\n")]


def generate_embeddings(
    lines: list[str],
    output_file: str,
    batch_size: int = 500,
) -> tuple[int, int]:
    model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    for i in tqdm(range(0, len(lines), batch_size), desc="Generating embeddings"):
        batch = lines[i:i + batch_size]
        batch_embeddings = list(model.embed(batch))

        if i == 0:
            dim = batch_embeddings[0].shape[0]
            mmap = np.memmap(output_file, dtype="float32", mode="w+", shape=(len(lines), dim))

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