from __future__ import annotations

import argparse
import asyncio
from pprint import pprint

from eduagentic.app import ConferenceEduSystem


async def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect available local models from configured endpoints.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config file.")
    args = parser.parse_args()
    system = ConferenceEduSystem(args.config)
    models = await system.registry.refresh(force=True)
    for endpoint, descriptors in models.items():
        print(f"\n[{endpoint}]")
        for descriptor in descriptors:
            pprint({"model_id": descriptor.model_id, "capability": descriptor.capability, "rank_key": descriptor.rank_key})


if __name__ == "__main__":
    asyncio.run(main())
