from __future__ import annotations

from eduagentic.retrieval.corpus import SourceDocument, chunk_documents
from eduagentic.retrieval.index import HybridIndex


def test_hybrid_index_returns_relevant_chunk():
    docs = [
        SourceDocument(doc_id="bio", title="Biology Notes", text="Mitosis is a process of cell division that produces two identical daughter cells."),
        SourceDocument(doc_id="phys", title="Physics Notes", text="Acceleration is the rate of change of velocity with respect to time."),
    ]
    chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=5)
    index = HybridIndex().fit(chunks)
    result = index.search("What does mitosis produce?", top_k=2)
    assert result.chunks
    assert result.chunks[0].doc_id == "bio"
