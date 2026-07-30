import argparse

from loganalyzer import generate_embeddings, load_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    emb = sub.add_parser("generate-embeddings")
    emb.add_argument("log_file")
    
    emb = sub.add_parser("teste")

    args = parser.parse_args()

    if args.command == "generate-embeddings":
        generate_embeddings(args.log_file)
    elif args.command == "teste":
        print("teste")
