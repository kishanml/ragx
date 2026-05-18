from __future__ import annotations
import argparse












def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragx",
        description="RAG experiment toolkit.",
    )
    parser.add_argument("--doc",action='store', help="Document's filepath",type=str)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ragx command-line interface."""
    parser = build_parser()
    document_fp = parser.parse_args(argv).doc
    
    
    
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
