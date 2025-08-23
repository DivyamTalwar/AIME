import json
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import hashlib
import os

import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
import anthropic
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIME")


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class AIMEMemoryNode:
    id: str
    content: str
    timestamp: datetime
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context: str = ""
    embedding: Optional[np.ndarray] = None
    links: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evolution_count: int = 0
    intelligence_score: float = 0.0
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        data['links'] = list(self.links)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AIMEMemoryNode':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data['embedding']:
            data['embedding'] = np.array(data['embedding'])
        data['links'] = set(data['links'])
        return cls(**data)


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        pass
    
    @abstractmethod
    async def generate_json(self, prompt: str, temperature: float = 0.3) -> Dict:
        pass


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_json(self, prompt: str, temperature: float = 0.3) -> Dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)


class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_json(self, prompt: str, temperature: float = 0.3) -> Dict:
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no other text."
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=temperature,
            messages=[{"role": "user", "content": json_prompt}]
        )
        return json.loads(response.content[0].text)


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return response.text
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_json(self, prompt: str, temperature: float = 0.3) -> Dict:
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no other text."
        response = await asyncio.to_thread(
            self.model.generate_content,
            json_prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return json.loads(response.text)


def create_llm(provider: LLMProvider, api_key: str, model: Optional[str] = None) -> BaseLLM:
    if provider == LLMProvider.OPENAI:
        return OpenAILLM(api_key, model or "gpt-4o-mini")
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicLLM(api_key, model or "claude-3-5-sonnet-20241022")
    elif provider == LLMProvider.GEMINI:
        return GeminiLLM(api_key, model or "gemini-1.5-flash")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class AIMEVectorEngine:
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
        
        self.index = self.pc.Index(index_name)
    
    def upsert(self, memory_id: str, embedding: np.ndarray, metadata: Dict = None):
        self.index.upsert(
            vectors=[{
                "id": memory_id,
                "values": embedding.tolist(),
                "metadata": metadata or {}
            }]
        )
    
    def query(self, embedding: np.ndarray, k: int = 10, filter: Dict = None) -> List[Tuple[str, float]]:
        results = self.index.query(
            vector=embedding.tolist(),
            top_k=k,
            filter=filter,
            include_metadata=False
        )
        return [(match["id"], match["score"]) for match in results["matches"]]
    
    def delete(self, memory_id: str):
        self.index.delete(ids=[memory_id])
    
    def delete_all(self):
        self.index.delete(delete_all=True)


class AIME:
    def __init__(
        self,
        llm_provider: LLMProvider,
        llm_api_key: str,
        pinecone_api_key: str,
        pinecone_environment: str,
        pinecone_index_name: str,
        pinecone_dimension: int = 384,
        pinecone_metric: str = "cosine",
        pinecone_cloud: str = "aws",
        pinecone_region: str = "us-east-1",
        llm_model: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        storage_path: Optional[Path] = None,
        max_links_per_memory: int = 5,
        evolution_threshold: float = 0.7,
        retrieval_k: int = 10,
        intelligence_threshold: float = 0.8
    ):
        self.llm = create_llm(llm_provider, llm_api_key, llm_model)
        self.encoder = SentenceTransformer(embedding_model, device='cpu')
        self.storage_path = Path(storage_path) if storage_path else Path("./aime_storage")
        self.storage_path.mkdir(exist_ok=True)
        
        self.vector_engine = AIMEVectorEngine(
            api_key=pinecone_api_key,
            environment=pinecone_environment,
            index_name=pinecone_index_name,
            dimension=pinecone_dimension,
            metric=pinecone_metric,
            cloud=pinecone_cloud,
            region=pinecone_region
        )
        
        self.memory_nodes: Dict[str, AIMEMemoryNode] = {}
        self.max_links = max_links_per_memory
        self.evolution_threshold = evolution_threshold
        self.retrieval_k = retrieval_k
        self.intelligence_threshold = intelligence_threshold
        
        self._load_memories()
        logger.info(f"AIME Engine initialized with {len(self.memory_nodes)} memories")
    
    async def encode_memory(
        self,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> AIMEMemoryNode:
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}
        
        memory_node = await self._construct_node(content, timestamp, metadata)
        await self._generate_neural_links(memory_node)
        await self._trigger_evolution(memory_node)
        
        self.memory_nodes[memory_node.id] = memory_node
        self.vector_engine.upsert(
            memory_node.id, 
            memory_node.embedding,
            {"timestamp": timestamp.isoformat(), "intelligence": memory_node.intelligence_score, **metadata}
        )
        
        self._save_memories()
        logger.info(f"AIME encoded memory {memory_node.id} | Links: {len(memory_node.links)} | Intelligence: {memory_node.intelligence_score:.2f}")
        return memory_node
    
    async def _construct_node(
        self,
        content: str,
        timestamp: datetime,
        metadata: Dict
    ) -> AIMEMemoryNode:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        node_id = f"aime_{timestamp.strftime('%Y%m%d_%H%M%S')}_{content_hash}"
        
        prompt = self._get_construction_prompt(content)
        attributes = await self.llm.generate_json(prompt)
        
        text_for_embedding = f"{content} {attributes.get('context', '')} {' '.join(attributes.get('keywords', []))} {' '.join(attributes.get('tags', []))}"
        embedding = self.encoder.encode(text_for_embedding, device='cpu', convert_to_numpy=True)
        
        intelligence_score = self._calculate_intelligence(attributes)
        
        return AIMEMemoryNode(
            id=node_id,
            content=content,
            timestamp=timestamp,
            keywords=attributes.get('keywords', []),
            tags=attributes.get('tags', []),
            context=attributes.get('context', ''),
            embedding=embedding,
            metadata=metadata,
            intelligence_score=intelligence_score
        )
    
    def _calculate_intelligence(self, attributes: Dict) -> float:
        keyword_score = min(len(attributes.get('keywords', [])) / 10, 1.0)
        tag_score = min(len(attributes.get('tags', [])) / 5, 1.0)
        context_score = min(len(attributes.get('context', '')) / 100, 1.0)
        return (keyword_score + tag_score + context_score) / 3
    
    async def _generate_neural_links(self, new_node: AIMEMemoryNode):
        if not self.memory_nodes:
            return
        
        nearest = self.vector_engine.query(new_node.embedding, k=self.retrieval_k)
        
        if not nearest:
            return
        
        nearest_nodes = [self.memory_nodes[node_id] for node_id, _ in nearest if node_id in self.memory_nodes]
        prompt = self._get_link_prompt(new_node, nearest_nodes)
        
        link_decision = await self.llm.generate_json(prompt)
        
        if link_decision.get('should_link') and link_decision.get('linked_memories'):
            for node_id in link_decision['linked_memories'][:self.max_links]:
                if node_id in self.memory_nodes:
                    new_node.links.add(node_id)
                    self.memory_nodes[node_id].links.add(new_node.id)
    
    async def _trigger_evolution(self, new_node: AIMEMemoryNode):
        if not new_node.links:
            return
        
        linked_nodes = [self.memory_nodes[node_id] for node_id in new_node.links if node_id in self.memory_nodes]
        
        for node in linked_nodes:
            prompt = self._get_evolution_prompt(new_node, node, linked_nodes)
            evolution = await self.llm.generate_json(prompt)
            
            if evolution.get('should_evolve'):
                if evolution.get('new_context'):
                    node.context = evolution['new_context']
                
                if evolution.get('new_tags'):
                    node.tags = list(set(node.tags + evolution['new_tags']))
                
                if evolution.get('new_keywords'):
                    node.keywords = list(set(node.keywords + evolution['new_keywords']))
                
                text_for_embedding = f"{node.content} {node.context} {' '.join(node.keywords)} {' '.join(node.tags)}"
                node.embedding = self.encoder.encode(text_for_embedding, device='cpu', convert_to_numpy=True)
                
                node.evolution_count += 1
                node.intelligence_score = min(node.intelligence_score * 1.1, 1.0)
                
                self.vector_engine.upsert(
                    node.id,
                    node.embedding,
                    {"timestamp": node.timestamp.isoformat(), "intelligence": node.intelligence_score, **node.metadata}
                )
                
                logger.info(f"AIME evolution triggered for {node.id} | Generation: {node.evolution_count} | Intelligence: {node.intelligence_score:.2f}")
    
    async def recall(
        self,
        query: str,
        k: int = None,
        include_neural_links: bool = True,
        traversal_depth: int = 2,
        intelligence_filter: float = None
    ) -> List[AIMEMemoryNode]:
        k = k or self.retrieval_k
        intelligence_filter = intelligence_filter or 0.0
        
        query_embedding = self.encoder.encode(query, device='cpu', convert_to_numpy=True)
        results = self.vector_engine.query(query_embedding, k=k)
        
        recalled_nodes = []
        visited = set()
        
        for node_id, score in results:
            if node_id in self.memory_nodes and node_id not in visited:
                node = self.memory_nodes[node_id]
                if node.intelligence_score >= intelligence_filter:
                    recalled_nodes.append(node)
                    visited.add(node_id)
                    
                    if include_neural_links:
                        self._traverse_neural_network(
                            node, recalled_nodes, visited, 
                            current_depth=0, max_depth=traversal_depth,
                            intelligence_filter=intelligence_filter
                        )
        
        return recalled_nodes
    
    def _traverse_neural_network(
        self,
        node: AIMEMemoryNode,
        collected: List[AIMEMemoryNode],
        visited: Set[str],
        current_depth: int,
        max_depth: int,
        intelligence_filter: float
    ):
        if current_depth >= max_depth:
            return
        
        for link_id in node.links:
            if link_id not in visited and link_id in self.memory_nodes:
                linked_node = self.memory_nodes[link_id]
                if linked_node.intelligence_score >= intelligence_filter:
                    collected.append(linked_node)
                    visited.add(link_id)
                    
                    self._traverse_neural_network(
                        linked_node, collected, visited,
                        current_depth + 1, max_depth,
                        intelligence_filter
                    )
    
    def get_neural_topology(self) -> Dict[str, List[str]]:
        topology = {}
        for node_id, node in self.memory_nodes.items():
            topology[node_id] = list(node.links)
        return topology
    
    def get_intelligence_metrics(self) -> Dict:
        if not self.memory_nodes:
            return {
                'total_nodes': 0,
                'neural_connections': 0,
                'avg_connections': 0,
                'max_evolution': 0,
                'avg_evolution': 0,
                'intelligent_nodes': 0,
                'network_intelligence': 0
            }
        
        total_links = sum(len(n.links) for n in self.memory_nodes.values())
        avg_links = total_links / len(self.memory_nodes)
        
        evolution_counts = [n.evolution_count for n in self.memory_nodes.values()]
        intelligence_scores = [n.intelligence_score for n in self.memory_nodes.values()]
        
        return {
            'total_nodes': len(self.memory_nodes),
            'neural_connections': total_links,
            'avg_connections': avg_links,
            'max_evolution': max(evolution_counts) if evolution_counts else 0,
            'avg_evolution': sum(evolution_counts) / len(evolution_counts) if evolution_counts else 0,
            'intelligent_nodes': sum(1 for n in self.memory_nodes.values() if n.intelligence_score >= self.intelligence_threshold),
            'network_intelligence': sum(intelligence_scores) / len(intelligence_scores) if intelligence_scores else 0,
            'evolved_nodes': sum(1 for n in self.memory_nodes.values() if n.evolution_count > 0)
        }
    
    def _get_construction_prompt(self, content: str) -> str:
        return f"""Analyze and extract intelligence from the following content:
1. Identify salient keywords (nouns, verbs, key concepts)
2. Extract core themes and contextual elements
3. Create categorical tags for classification

Format the response as JSON:
{{
    "keywords": ["specific, distinct keywords capturing key concepts"],
    "context": "one sentence summarizing main topic, key points, and purpose",
    "tags": ["broad categories/themes for classification"]
}}

Content for analysis:
{content}"""
    
    def _get_link_prompt(
        self,
        new_node: AIMEMemoryNode,
        nearest_nodes: List[AIMEMemoryNode]
    ) -> str:
        nearest_info = "\n".join([
            f"- ID: {n.id}, Context: {n.context}, Intelligence: {n.intelligence_score:.2f}"
            for n in nearest_nodes[:5]
        ])
        
        return f"""Analyze neural connections for memory integration.

New memory node:
- Context: {new_node.context}
- Content: {new_node.content[:200]}...
- Keywords: {new_node.keywords}
- Intelligence: {new_node.intelligence_score:.2f}

Candidate nodes for connection:
{nearest_info}

Determine neural links based on semantic relationships and conceptual connections.

Return JSON:
{{
    "should_link": true/false,
    "linked_memories": ["node_id1", "node_id2", ...],
    "reasoning": "brief explanation"
}}"""
    
    def _get_evolution_prompt(
        self,
        new_node: AIMEMemoryNode,
        target_node: AIMEMemoryNode,
        related_nodes: List[AIMEMemoryNode]
    ) -> str:
        related_info = "\n".join([
            f"- Context: {n.context[:100]}..."
            for n in related_nodes[:3] if n.id != target_node.id
        ])
        
        return f"""Determine memory evolution based on new intelligence.

New intelligence context: {new_node.context}
New intelligence keywords: {new_node.keywords}

Target memory for evolution:
- ID: {target_node.id}
- Context: {target_node.context}
- Tags: {target_node.tags}
- Keywords: {target_node.keywords}
- Evolution generation: {target_node.evolution_count}
- Intelligence score: {target_node.intelligence_score:.2f}

Related neural context:
{related_info}

Determine if the target should evolve to incorporate new intelligence.

Return JSON:
{{
    "should_evolve": true/false,
    "new_context": "updated context if applicable" or null,
    "new_tags": ["additional tags"] or [],
    "new_keywords": ["additional keywords"] or [],
    "reasoning": "brief explanation"
}}"""
    
    def _save_memories(self):
        import pickle
        save_path = self.storage_path / "aime_memories.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump({
                'memory_nodes': {k: v.to_dict() for k, v in self.memory_nodes.items()}
            }, f)
    
    def _load_memories(self):
        import pickle
        save_path = self.storage_path / "aime_memories.pkl"
        if not save_path.exists():
            return
        
        try:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
            
            for node_id, node_dict in data['memory_nodes'].items():
                node = AIMEMemoryNode.from_dict(node_dict)
                self.memory_nodes[node_id] = node
            
            logger.info(f"AIME loaded {len(self.memory_nodes)} memory nodes from storage")
        except Exception as e:
            logger.error(f"Failed to load AIME memories: {e}")


class AIMEAgent:
    def __init__(self, aime_engine: AIME):
        self.aime = aime_engine
        self.conversation_history = []
    
    async def process(self, user_input: str) -> str:
        relevant_memories = await self.aime.recall(user_input, k=5)
        memory_context = self._build_context(relevant_memories)
        
        prompt = f"""You are an intelligent agent powered by AIME (Agentic Intelligent Memory Engine).

Neural memory context:
{memory_context}

Current input: {user_input}

Respond intelligently using relevant memories when appropriate."""
        
        response = await self.aime.llm.generate(prompt)
        
        interaction = f"User: {user_input}\nAIME: {response}"
        await self.aime.encode_memory(interaction)
        
        self.conversation_history.append({
            'user': user_input,
            'aime': response,
            'timestamp': datetime.now()
        })
        
        return response
    
    def _build_context(self, memories: List[AIMEMemoryNode]) -> str:
        if not memories:
            return "No relevant neural memories found."
        
        context_parts = []
        for memory in memories[:3]:
            context_parts.append(
                f"[{memory.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"Intelligence: {memory.intelligence_score:.2f} | "
                f"Evolution: Gen {memory.evolution_count}\n"
                f"Context: {memory.context}\n"
                f"Content: {memory.content[:200]}..."
            )
        
        return "\n\n".join(context_parts)


async def aime_demo():
    from dotenv import load_dotenv
    load_dotenv()
    
    aime_engine = AIME(
        llm_provider=LLMProvider.OPENAI,
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "aime-neural-network"),
        pinecone_dimension=int(os.getenv("PINECONE_DIMENSION", "384")),
        pinecone_metric=os.getenv("PINECONE_METRIC", "cosine"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1")
    )
    
    agent = AIMEAgent(aime_engine)
    
    conversations = [
        "I'm developing an AI system that needs to learn from experience.",
        "The system should evolve its understanding over time, like a neural network.",
        "How can we implement memory that actually grows more intelligent?",
        "What if memories could form their own connections autonomously?",
        "Tell me about the AI system we're building - how does it learn?"
    ]
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     AIME — Agentic Intelligent Memory Engine Demo       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    for i, message in enumerate(conversations, 1):
        print(f"Session {i}:")
        print(f"Human: {message}")
        
        response = await agent.process(message)
        print(f"AIME: {response}\n")
        
        metrics = aime_engine.get_intelligence_metrics()
        print(f"Neural Metrics: Nodes: {metrics['total_nodes']} | Connections: {metrics['neural_connections']} | Intelligence: {metrics['network_intelligence']:.2f}\n")
        print("-" * 60 + "\n")
        
        await asyncio.sleep(1)
    
    print("=== AIME Neural Network Analysis ===\n")
    
    test_queries = [
        "How does the AI system evolve?",
        "What makes memories intelligent?",
        "Explain the neural connections"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        memories = await aime_engine.recall(query, k=3, intelligence_filter=0.5)
        print(f"Recalled {len(memories)} intelligent memories:")
        for mem in memories:
            print(f"  • Intelligence: {mem.intelligence_score:.2f} | {mem.context[:80]}...")
        print()
    
    topology = aime_engine.get_neural_topology()
    print(f"\n=== AIME Neural Topology ===")
    print(f"Total neural nodes: {len(topology)}")
    print(f"Interconnected nodes: {sum(1 for links in topology.values() if links)}")
    
    return aime_engine


if __name__ == "__main__":
    asyncio.run(aime_demo())