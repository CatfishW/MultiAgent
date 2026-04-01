from __future__ import annotations

import argparse
import asyncio

from eduagentic.app import ConferenceEduSystem


async def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke demo against the configured local endpoints.")
    parser.add_argument("question")
    parser.add_argument("--config", default=None)
    parser.add_argument("--architecture", default=None)
    args = parser.parse_args()

    system = ConferenceEduSystem(args.config)
    response = await system.answer(args.question, architecture=args.architecture)
    print(response.answer)
    print("\nArchitecture:", response.architecture.value)
    print("Route scores:", response.route.scores)
    if response.citations:
        print("Citations:", response.citations)


if __name__ == "__main__":
    asyncio.run(main())
