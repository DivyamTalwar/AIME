import pytest
from datetime import datetime
import numpy as np
from agenticMemory import AIMEMemoryNode

@pytest.fixture
def memory_node_fixture():
    """Provides a sample AIMEMemoryNode for testing."""
    timestamp = datetime.now()
    embedding = np.array([0.1, 0.2, 0.3])
    return AIMEMemoryNode(
        id="test_node_123",
        content="This is a test memory.",
        timestamp=timestamp,
        keywords=["test", "memory"],
        tags=["testing", "pytest"],
        context="A node used for unit testing.",
        embedding=embedding,
        links={"linked_node_1", "linked_node_2"},
        metadata={"source": "pytest"},
        evolution_count=1,
        intelligence_score=0.75
    )

def test_memory_node_creation(memory_node_fixture):
    """Tests that a memory node is created with the correct attributes."""
    node = memory_node_fixture
    assert node.id == "test_node_123"
    assert node.content == "This is a test memory."
    assert isinstance(node.timestamp, datetime)
    assert node.keywords == ["test", "memory"]
    assert node.tags == ["testing", "pytest"]
    assert node.context == "A node used for unit testing."
    np.testing.assert_array_equal(node.embedding, np.array([0.1, 0.2, 0.3]))
    assert node.links == {"linked_node_1", "linked_node_2"}
    assert node.metadata == {"source": "pytest"}
    assert node.evolution_count == 1
    assert node.intelligence_score == 0.75

def test_to_dict_serialization(memory_node_fixture):
    """Tests the serialization of a memory node to a dictionary."""
    node = memory_node_fixture
    node_dict = node.to_dict()

    assert node_dict['id'] == node.id
    assert node_dict['timestamp'] == node.timestamp.isoformat()
    assert node_dict['embedding'] == [0.1, 0.2, 0.3]
    assert isinstance(node_dict['links'], list)
    assert set(node_dict['links']) == {"linked_node_1", "linked_node_2"}
    assert node_dict['intelligence_score'] == 0.75

def test_from_dict_deserialization(memory_node_fixture):
    """Tests the deserialization of a dictionary back to a memory node."""
    node = memory_node_fixture
    node_dict = node.to_dict()

    rehydrated_node = AIMEMemoryNode.from_dict(node_dict)

    assert rehydrated_node.id == node.id
    assert rehydrated_node.content == node.content
    assert rehydrated_node.timestamp == node.timestamp
    assert rehydrated_node.keywords == node.keywords
    assert rehydrated_node.tags == node.tags
    assert rehydrated_node.context == node.context
    np.testing.assert_array_equal(rehydrated_node.embedding, node.embedding)
    assert rehydrated_node.links == node.links
    assert rehydrated_node.metadata == node.metadata
    assert rehydrated_node.evolution_count == node.evolution_count
    assert rehydrated_node.intelligence_score == node.intelligence_score

def test_serialization_deserialization_roundtrip(memory_node_fixture):
    """Ensures that a node remains unchanged after a to_dict/from_dict roundtrip."""
    original_node = memory_node_fixture
    node_dict = original_node.to_dict()
    rehydrated_node = AIMEMemoryNode.from_dict(node_dict)

    assert original_node == rehydrated_node
    np.testing.assert_array_equal(original_node.embedding, rehydrated_node.embedding)

def test_node_with_empty_fields():
    """Tests a node with default or empty fields."""
    timestamp = datetime.now()
    node = AIMEMemoryNode(id="empty_node", content="Empty", timestamp=timestamp)

    assert node.keywords == []
    assert node.tags == []
    assert node.context == ""
    assert node.embedding is None
    assert node.links == set()
    assert node.metadata == {}
    assert node.evolution_count == 0
    assert node.intelligence_score == 0.0

    # Test roundtrip with empty fields
    node_dict = node.to_dict()
    rehydrated_node = AIMEMemoryNode.from_dict(node_dict)
    assert node == rehydrated_node