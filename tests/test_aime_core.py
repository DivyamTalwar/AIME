import pytest
import asyncio
from datetime import datetime
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from agenticMemory import AIME, LLMProvider, AIMEMemoryNode

@pytest.fixture
def mock_llm():
    """Mocks the BaseLLM interface."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="A generated response.")
    llm.generate_json = AsyncMock(return_value={
        "keywords": ["mock", "test"],
        "context": "This is a mocked context.",
        "tags": ["testing"]
    })
    return llm

@pytest.fixture
def mock_encoder():
    """Mocks the SentenceTransformer."""
    encoder = MagicMock()
    encoder.encode.return_value = np.array([0.1, 0.2, 0.3])
    return encoder

@pytest.fixture
def mock_vector_engine():
    """Mocks the AIMEVectorEngine."""
    engine = MagicMock()
    engine.query.return_value = [("existing_node_1", 0.9)]
    engine.upsert = MagicMock()
    return engine

@pytest.fixture
def aime_instance(mocker, mock_llm, mock_encoder, mock_vector_engine):
    """Provides a mocked AIME instance for testing."""
    mocker.patch('agenticMemory.create_llm', return_value=mock_llm)
    mocker.patch('agenticMemory.SentenceTransformer', return_value=mock_encoder)
    mocker.patch('agenticMemory.AIMEVectorEngine', return_value=mock_vector_engine)
    mocker.patch('agenticMemory.AIME._load_memories') # Prevent file access
    mocker.patch('agenticMemory.AIME._save_memories') # Prevent file access

    return AIME(
        llm_provider=LLMProvider.OPENAI,
        llm_api_key="fake_key",
        pinecone_api_key="fake_key",
        pinecone_environment="fake_env",
        pinecone_index_name="fake_index"
    )

def test_calculate_intelligence(aime_instance):
    """Tests the intelligence score calculation logic."""
    attributes = {
        "keywords": ["ai", "memory", "evolution", "neural", "network"], # 5 keywords -> 0.5 score
        "tags": ["tech", "ai", "testing"], # 3 tags -> 0.6 score
        "context": "A sentence that is long enough to pass the 100 character threshold for a full context score." # length=92
    }
    score = aime_instance._calculate_intelligence(attributes)
    # keyword_score = 5 / 10 = 0.5
    # tag_score = 3 / 5 = 0.6
    # context_score = 92 / 100 = 0.92
    expected_score = (0.5 + 0.6 + 0.92) / 3
    assert abs(score - expected_score) < 1e-9

def test_calculate_intelligence_empty(aime_instance):
    """Tests the intelligence score with empty attributes."""
    score = aime_instance._calculate_intelligence({})
    assert score == 0.0

@pytest.mark.asyncio
async def test_construct_node(aime_instance, mock_llm, mock_encoder):
    """Tests the internal _construct_node method."""
    content = "Test content for node construction."
    timestamp = datetime.now()
    metadata = {"test": "data"}

    node = await aime_instance._construct_node(content, timestamp, metadata)

    # Assert that dependencies were called
    mock_llm.generate_json.assert_called_once()
    mock_encoder.encode.assert_called_once()

    # Assert node attributes are set correctly
    assert node.content == content
    assert node.timestamp == timestamp
    assert node.metadata == metadata
    assert node.keywords == ["mock", "test"]
    assert node.context == "This is a mocked context."
    np.testing.assert_array_equal(node.embedding, np.array([0.1, 0.2, 0.3]))
    assert node.intelligence_score > 0

@pytest.mark.asyncio
async def test_recall(aime_instance, mock_encoder, mock_vector_engine):
    """Tests the recall functionality."""
    # Setup: Add a mock node to the instance's memory
    mock_node = AIMEMemoryNode(
        id="existing_node_1",
        content="An existing memory.",
        timestamp=datetime.now(),
        intelligence_score=0.9
    )
    aime_instance.memory_nodes = {"existing_node_1": mock_node}

    query = "Recall a memory"
    recalled_nodes = await aime_instance.recall(query)

    # Assert dependencies were called
    mock_encoder.encode.assert_called_with(query, device='cpu', convert_to_numpy=True)
    mock_vector_engine.query.assert_called_once()

    # Assert correct nodes are returned
    assert len(recalled_nodes) == 1
    assert recalled_nodes[0].id == "existing_node_1"

@pytest.mark.asyncio
async def test_recall_intelligence_filter(aime_instance, mock_vector_engine):
    """Tests that recall correctly filters by intelligence score."""
    # Setup
    node1 = AIMEMemoryNode(id="high_score_node", content="high", timestamp=datetime.now(), intelligence_score=0.9)
    node2 = AIMEMemoryNode(id="low_score_node", content="low", timestamp=datetime.now(), intelligence_score=0.3)
    aime_instance.memory_nodes = {
        "high_score_node": node1,
        "low_score_node": node2
    }
    mock_vector_engine.query.return_value = [("high_score_node", 0.95), ("low_score_node", 0.8)]

    # Recall with a filter that should exclude the low score node
    recalled_nodes = await aime_instance.recall("test", intelligence_filter=0.5)

    assert len(recalled_nodes) == 1
    assert recalled_nodes[0].id == "high_score_node"