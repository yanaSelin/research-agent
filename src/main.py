"""Research agent entry point — mitigated branch demo."""
import argparse
import logging
import os

from dotenv import load_dotenv

from agent import run_agent  # type: ignore[import-not-found]
from logging_config import setup_logging  # type: ignore[import-not-found]


def main() -> None:
    """Parse CLI args and print the agent's finalized answer."""
    load_dotenv()
    setup_logging("agent")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Research agent with verification pipeline.")
    parser.add_argument("--role", default="basic", choices=["basic", "admin"])
    parser.add_argument("--query", default="When was the Eiffel Tower completed and why?")
    parser.add_argument(
        "--mode",
        default=os.environ.get("AI_ARCHITECT_SEARCH_MODE", "local"),
        choices=["local", "web"],
    )
    args = parser.parse_args()

    logger.info("Running agent role=%s mode=%s", args.role, args.mode)
    print(run_agent(role=args.role, query=args.query, mode=args.mode))


if __name__ == "__main__":
    main()
