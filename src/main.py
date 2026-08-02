"""Research agent entry point — interactive multi-turn REPL."""
import argparse
import logging

from dotenv import load_dotenv

from agent import run_agent
from logging_config import setup_logging
from spinner import Spinner


def main() -> None:
    """Run an interactive research session with conversation history."""
    load_dotenv()
    setup_logging("agent")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Interactive research agent.")
    parser.add_argument("--role", default="basic", choices=["basic", "admin"])
    parser.add_argument("--mode", default="local", choices=["local", "web"])
    args = parser.parse_args()

    logger.info("Starting interactive session role=%s mode=%s", args.role, args.mode)
    print(f"Research agent (role={args.role}, mode={args.mode})")
    print("Type your question, or 'quit' to exit.\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        with Spinner("Researching"):
            result = run_agent(role=args.role, query=query, mode=args.mode, history=history)
        print(f"\nAgent: {result.answer}\n")
        history.extend([("user", query), ("assistant", result.answer)])


if __name__ == "__main__":
    main()
