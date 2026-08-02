"""Research agent entry point — mitigated branch demo."""
import argparse
import logging

from dotenv import load_dotenv

from agent import run_agent
from logging_config import setup_logging
from spinner import Spinner


def main() -> None:
    """Parse CLI args and print the agent's finalized answer."""
    load_dotenv()
    setup_logging("agent")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Research agent with verification pipeline.")
    parser.add_argument("--role", default="basic", choices=["basic", "admin"])
    parser.add_argument("--query", default="When was the Eiffel Tower completed and why?")
    args = parser.parse_args()

    logger.info("Running agent role=%s", args.role)
    with Spinner("Researching"):
        answer = run_agent(role=args.role, query=args.query, mode="local")
    print(answer)


if __name__ == "__main__":
    main()
