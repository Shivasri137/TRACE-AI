from core import retrieve_evidence, split_claims, split_into_chunks

def test_split_claims():
    assert len(split_claims("The study enrolled 120 participants. The trial lasted eight weeks.")) == 2

def test_split_empty_text():
    assert split_into_chunks("") == []

def test_chunking_creates_multiple_passages():
    chunks = split_into_chunks("Research evidence. " * 200, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)

def test_retrieval_ranks_relevant_passage():
    chunks = [
        "The experiment used a neural network for image classification.",
        "The capital of France is Paris.",
        "The neural network classified images using training data.",
    ]
    results = retrieve_evidence("The neural network classified images.", chunks, top_k=2)
    assert len(results) == 2
    assert results[0]["relevance"] >= results[1]["relevance"]
    assert "neural network" in results[0]["text"]

def test_retrieval_empty_input():
    assert retrieve_evidence("test", []) == []
    assert retrieve_evidence("", ["some evidence"]) == []

def test_invalid_chunk_parameters():
    import pytest
    with pytest.raises(ValueError):
        split_into_chunks("text", chunk_size=10, overlap=10)
