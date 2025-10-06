import pytest
import asyncio
from datetime import datetime
import numpy as np
from unittest.mock import MagicMock, AsyncMock, call

from agenticMemory import AIME, LLMProvider, AIMEMemoryNode

@pytest.fixture
def mock_llm_integration():
    """
    Mocks the BaseLLM for a multi-step integration test.
    It uses side_effect to provide different responses for each call.
    """
    llm = MagicMock()

    # Define the sequence of JSON responses
    json_responses = [
        # 1. Construction of the first node
        {
            "keywords": ["initial", "memory"],
            "context": "This is the original memory about a topic.",
            "tags": ["base"]
        },
        # 2. Construction of the second, related node
        {
            "keywords": ["new", "intelligence"],
            "context": "This is new intelligence related to the original topic.",
            "tags": ["update"]
        },
        # 3. Linking decision: link the new node to the first one
        {
            "should_link": True,
            "linked_memories": ["aime_20250101_000000_initial"], # The ID of the first node
            "reasoning": "They are semantically related."
        },
        # 4. Evolution decision: evolve the first node based on the new one
        {
            "should_evolve": True,
            "new_context": "The original memory has now been evolved with new intelligence.",
            "new_tags": ["evolved"],
            "new_keywords": ["updated"],
            "reasoning": "The new information expands upon the original."
        }
    ]

    llm.generate_json = AsyncMock(side_effect=json_responses)
    return llm

@pytest.fixture
def mock_encoder_integration():
    """Mocks the SentenceTransformer to provide distinct embeddings."""
    encoder = MagicMock()
    encoder.encode.side_effect = [
        np.array([0.1] * 384), # 1. Embedding for the first node
        np.array([0.2] * 384), # 2. Embedding for the second node
        np.array([0.3] * 384)  # 3. Embedding for the evolved first node
    ]
    return encoder

@pytest.fixture
def mock_vector_engine_integration(mocker):
    """Mocks the AIMEVectorEngine for integration flow."""
    engine = MagicMock()
    # When queried with the second node's embedding, it should find the first node.
    engine.query.return_value = [("aime_20250101_000000_initial", 0.95)]
    engine.upsert = mocker.MagicMock()
    return engine

@pytest.fixture
def aime_integration_instance(mocker, mock_llm_integration, mock_encoder_integration, mock_vector_engine_integration):
    """Provides a mocked AIME instance for integration testing."""
    mocker.patch('agenticMemory.create_llm', return_value=mock_llm_integration)
    mocker.patch('agenticMemory.SentenceTransformer', return_value=mock_encoder_integration)
    mocker.patch('agenticMemory.AIMEVectorEngine', return_value=mock_vector_engine_integration)
    mocker.patch('agenticMemory.AIME._load_memories')
    mocker.patch('agenticMemory.AIME._save_memories')

    # Patch hashlib to control the generated node IDs for predictability
    mocker.patch('hashlib.md5', side_effect=[
        MagicMock(hexdigest=lambda: "initial"), # for the first node
        MagicMock(hexdigest=lambda: "new")      # for the second node
    ])

    return AIME(
        llm_provider=LLMProvider.OPENAI,
        llm_api_key="fake_key",
        pinecone_api_key="fake_key",
        pinecone_environment="fake_env",
        pinecone_index_name="fake_index"
    )

@pytest.mark.asyncio
async def test_memory_encoding_and_evolution_flow(aime_integration_instance, mock_llm_integration, mock_vector_engine_integration):
    """
    Tests the full flow:
    1. Encode a memory.
    2. Encode a related memory.
    3. Verify that the first memory is linked and evolves.
    """
    aime = aime_integration_instance

    # 1. Encode the first (foundational) memory
    # We need a fixed timestamp to predict the generated ID
    timestamp1 = datetime(2025, 1, 1, 0, 0, 0)
    node1 = await aime.encode_memory(
        content="Original memory content.",
        timestamp=timestamp1
    )

    # Verify the first node's initial state
    assert node1.id == "aime_20250101_000000_initial"
    assert node1.context == "This is the original memory about a topic."
    assert node1.evolution_count == 0
    initial_intelligence = node1.intelligence_score

    # 2. Encode the second (related) memory
    timestamp2 = datetime(2025, 1, 1, 0, 1, 0)
    node2 = await aime.encode_memory(
        content="New related intelligence.",
        timestamp=timestamp2
    )

    # Verify the second node's state
    assert node2.id == "aime_20250101_000100_new"
    assert node2.context == "This is new intelligence related to the original topic."

    # 3. Verify the linking and evolution of the first node
    evolved_node1 = aime.memory_nodes[node1.id]

    # Check that linking occurred
    assert node2.id in evolved_node1.links
    assert evolved_node1.id in node2.links

    # Check that evolution occurred
    assert evolved_node1.evolution_count == 1
    assert evolved_node1.intelligence_score > initial_intelligence
    assert evolved_node1.context == "The original memory has now been evolved with new intelligence."
    assert "evolved" in evolved_node1.tags
    assert "updated" in evolved_node1.keywords

    # Check that the vector database was updated for the evolved node
    # Call order: 1. node1 initial upsert, 2. node1 evolution upsert, 3. node2 initial upsert
    assert mock_vector_engine_integration.upsert.call_count == 3

    # The evolution upsert is the *second* call, triggered during the creation of node2
    evolution_upsert_call = mock_vector_engine_integration.upsert.call_args_list[1]
    assert evolution_upsert_call[0][0] == evolved_node1.id # args[0] is the node ID

    # Check that the LLM was called the correct number of times with expected prompts
    assert mock_llm_integration.generate_json.call_count == 4

    # Optional: More detailed prompt checking if needed
    # link_prompt_call = mock_llm_integration.generate_json.call_args_list[2]
    # assert "Analyze neural connections" in link_prompt_call[0][0]
    # evolution_prompt_call = mock_llm_integration.generate_json.call_args_list[3]
    # assert "Determine memory evolution" in evolution_prompt_call[0][0]