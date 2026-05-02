"""Domain-configurable multi-agent retrieval orchestration framework.

This package extends the uploaded `agent_swarm_port` substrate with:
- fast local-LLM routing against OpenAI-compatible endpoints
- strong classical RAG and agentic RAG baselines
- coordination-heavy response, criteria, and planning pipelines
- benchmark adapters covering the dataset families listed in the paper
- lightweight trainable routing and reranking modules
"""

from .app import ConferenceEduSystem
from .config import AppConfig, load_app_config

MARLETSystem = ConferenceEduSystem

__all__ = ["ConferenceEduSystem", "MARLETSystem", "AppConfig", "load_app_config"]
