"""
Semantic Cache System for AI Assistant.

This module provides intelligent caching for LLM responses using semantic
similarity matching, reducing costs and improving response times for
similar queries.

Features:
- Semantic similarity matching using embeddings
- TTL-based cache expiration
- LRU eviction policy
- Persistent storage options (memory, SQLite, Redis)
- Cache statistics and monitoring
"""

import hashlib
import json
import time
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import OrderedDict
from pathlib import Path
import numpy as np


@dataclass
class CacheEntry:
    """
    Represents a cached LLM response.

    Attributes:
        key: Unique identifier for the cache entry
        prompt: Original prompt text
        response: Cached LLM response
        model: Model used to generate the response
        embedding: Optional embedding vector for semantic matching
        created_at: Timestamp when entry was created
        expires_at: Timestamp when entry expires
        hit_count: Number of times this entry was retrieved
        metadata: Additional metadata about the entry
    """
    key: str
    prompt: str
    response: str
    model: str
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "key": self.key,
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create entry from dictionary."""
        return cls(**data)


@dataclass
class CacheStats:
    """
    Cache performance statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        semantic_hits: Number of semantic similarity hits
        evictions: Number of cache evictions
        total_requests: Total number of cache requests
        avg_hit_latency_ms: Average latency for cache hits
        estimated_cost_saved: Estimated cost saved by caching
    """
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    evictions: int = 0
    total_requests: int = 0
    avg_hit_latency_ms: float = 0.0
    estimated_cost_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits + self.semantic_hits) / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "evictions": self.evictions,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "avg_hit_latency_ms": self.avg_hit_latency_ms,
            "estimated_cost_saved": self.estimated_cost_saved
        }


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry from the cache."""
        pass

    @abstractmethod
    def set(self, entry: CacheEntry) -> None:
        """Store an entry in the cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an entry from the cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass

    @abstractmethod
    def get_all_entries(self) -> List[CacheEntry]:
        """Get all entries in the cache."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get the number of entries in the cache."""
        pass


class MemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend with LRU eviction.

    This backend stores cache entries in memory using an OrderedDict
    for efficient LRU eviction.

    Args:
        max_size: Maximum number of entries to store

    Example:
        >>> backend = MemoryCacheBackend(max_size=1000)
        >>> backend.set(cache_entry)
        >>> entry = backend.get("key")
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry, moving it to the end (most recently used)."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            return entry

    def set(self, entry: CacheEntry) -> None:
        """Store an entry, evicting oldest if necessary."""
        with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[entry.key] = entry
            self._cache.move_to_end(entry.key)

    def delete(self, key: str) -> bool:
        """Delete an entry from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def get_all_entries(self) -> List[CacheEntry]:
        """Get all entries in the cache."""
        with self._lock:
            return list(self._cache.values())

    def size(self) -> int:
        """Get the number of entries in the cache."""
        with self._lock:
            return len(self._cache)


class SQLiteCacheBackend(CacheBackend):
    """
    SQLite-based persistent cache backend.

    This backend stores cache entries in a SQLite database for
    persistence across sessions.

    Args:
        db_path: Path to the SQLite database file
        max_size: Maximum number of entries to store

    Example:
        >>> backend = SQLiteCacheBackend("/tmp/cache.db")
        >>> backend.set(cache_entry)
    """

    def __init__(self, db_path: str = ":memory:", max_size: int = 10000):
        self.db_path = db_path
        self.max_size = max_size
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding TEXT,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    hit_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    last_accessed REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON cache_entries(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed
                ON cache_entries(last_accessed)
            """)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry from the database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                entry = self._row_to_entry(row)

                # Check expiration
                if entry.is_expired():
                    self.delete(key)
                    return None

                # Update hit count and last accessed
                conn.execute(
                    """UPDATE cache_entries
                       SET hit_count = hit_count + 1, last_accessed = ?
                       WHERE key = ?""",
                    (time.time(), key)
                )
                conn.commit()
                entry.hit_count += 1
                return entry

    def set(self, entry: CacheEntry) -> None:
        """Store an entry in the database."""
        with self._lock:
            with self._get_connection() as conn:
                # Enforce max size with LRU eviction
                count = conn.execute(
                    "SELECT COUNT(*) FROM cache_entries"
                ).fetchone()[0]

                while count >= self.max_size:
                    conn.execute("""
                        DELETE FROM cache_entries
                        WHERE key = (
                            SELECT key FROM cache_entries
                            ORDER BY last_accessed ASC LIMIT 1
                        )
                    """)
                    count -= 1

                embedding_json = (
                    json.dumps(entry.embedding)
                    if entry.embedding else None
                )
                metadata_json = json.dumps(entry.metadata)

                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (key, prompt, response, model, embedding, created_at,
                     expires_at, hit_count, metadata, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key, entry.prompt, entry.response, entry.model,
                    embedding_json, entry.created_at, entry.expires_at,
                    entry.hit_count, metadata_json, time.time()
                ))
                conn.commit()

    def delete(self, key: str) -> bool:
        """Delete an entry from the database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE key = ?",
                    (key,)
                )
                conn.commit()
                return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all entries from the database."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()

    def get_all_entries(self) -> List[CacheEntry]:
        """Get all entries in the database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM cache_entries")
                return [self._row_to_entry(row) for row in cursor.fetchall()]

    def size(self) -> int:
        """Get the number of entries in the database."""
        with self._lock:
            with self._get_connection() as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM cache_entries"
                ).fetchone()[0]

    def _row_to_entry(self, row: Tuple) -> CacheEntry:
        """Convert a database row to a CacheEntry."""
        return CacheEntry(
            key=row[0],
            prompt=row[1],
            response=row[2],
            model=row[3],
            embedding=json.loads(row[4]) if row[4] else None,
            created_at=row[5],
            expires_at=row[6],
            hit_count=row[7],
            metadata=json.loads(row[8]) if row[8] else {}
        )


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class SimpleHashEmbedding(EmbeddingProvider):
    """
    Simple hash-based pseudo-embedding for testing.

    This is not a real embedding - use for testing only.
    For production, use a proper embedding model.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        """Generate pseudo-embedding from text hash."""
        # Create a deterministic pseudo-embedding from text
        hash_bytes = hashlib.sha256(text.lower().encode()).digest()
        # Convert to floats
        values = []
        for i in range(0, min(len(hash_bytes), self.dimension), 4):
            chunk = hash_bytes[i:i+4]
            value = int.from_bytes(chunk, 'big') / (2**32)
            values.append(value * 2 - 1)  # Normalize to [-1, 1]

        # Pad or truncate to dimension
        while len(values) < self.dimension:
            values.append(0.0)

        return values[:self.dimension]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate pseudo-embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Google Gemini embedding provider.

    Uses Gemini's text-embedding model for semantic embeddings.

    Args:
        api_key: Google AI API key
        model: Embedding model name
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004"
    ):
        self.api_key = api_key
        self.model = model
        self._genai = None

    def _get_client(self):
        """Lazy load the Gemini client."""
        if self._genai is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
        return self._genai

    def embed(self, text: str) -> List[float]:
        """Generate embedding using Gemini."""
        genai = self._get_client()
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


class SemanticCache:
    """
    Semantic cache for LLM responses.

    This cache uses semantic similarity to match similar prompts,
    allowing cache hits even when prompts are not identical.

    Args:
        backend: Cache backend to use (memory, sqlite, etc.)
        embedding_provider: Provider for generating embeddings
        similarity_threshold: Minimum similarity score for cache hit (0-1)
        ttl_seconds: Time-to-live for cache entries in seconds
        enable_semantic_matching: Whether to use semantic similarity
        cost_per_1k_tokens: Estimated cost per 1K tokens for cost tracking

    Example:
        >>> cache = SemanticCache(
        ...     backend=MemoryCacheBackend(max_size=1000),
        ...     similarity_threshold=0.95
        ... )
        >>>
        >>> # Try to get cached response
        >>> result = cache.get("What is Apache Spark?", model="claude")
        >>> if result is None:
        ...     response = call_llm("What is Apache Spark?")
        ...     cache.set("What is Apache Spark?", response, model="claude")
    """

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        similarity_threshold: float = 0.95,
        ttl_seconds: Optional[int] = 3600,
        enable_semantic_matching: bool = True,
        cost_per_1k_tokens: float = 0.01
    ):
        self.backend = backend or MemoryCacheBackend()
        self.embedding_provider = embedding_provider or SimpleHashEmbedding()
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.enable_semantic_matching = enable_semantic_matching
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def _generate_key(self, prompt: str, model: str) -> str:
        """Generate a cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4

    def get(
        self,
        prompt: str,
        model: str,
        system_instruction: Optional[str] = None
    ) -> Optional[str]:
        """
        Get cached response for a prompt.

        Args:
            prompt: The prompt to look up
            model: The model name
            system_instruction: Optional system instruction

        Returns:
            Cached response if found, None otherwise
        """
        start_time = time.time()

        with self._lock:
            self.stats.total_requests += 1

            # Build full prompt for caching
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"[SYSTEM]{system_instruction}[/SYSTEM]{prompt}"

            # Try exact match first
            key = self._generate_key(full_prompt, model)
            entry = self.backend.get(key)

            if entry is not None:
                self.stats.hits += 1
                latency = (time.time() - start_time) * 1000
                self._update_latency(latency)
                self._update_cost_saved(entry.response)
                return entry.response

            # Try semantic matching if enabled
            if self.enable_semantic_matching:
                result = self._semantic_search(full_prompt, model)
                if result is not None:
                    self.stats.semantic_hits += 1
                    latency = (time.time() - start_time) * 1000
                    self._update_latency(latency)
                    self._update_cost_saved(result)
                    return result

            self.stats.misses += 1
            return None

    def set(
        self,
        prompt: str,
        response: str,
        model: str,
        system_instruction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a response in the cache.

        Args:
            prompt: The original prompt
            response: The LLM response
            model: The model name
            system_instruction: Optional system instruction
            metadata: Additional metadata to store
        """
        with self._lock:
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"[SYSTEM]{system_instruction}[/SYSTEM]{prompt}"

            key = self._generate_key(full_prompt, model)

            # Generate embedding for semantic matching
            embedding = None
            if self.enable_semantic_matching:
                try:
                    embedding = self.embedding_provider.embed(full_prompt)
                except Exception:
                    pass  # Continue without embedding

            # Calculate expiration
            expires_at = None
            if self.ttl_seconds:
                expires_at = time.time() + self.ttl_seconds

            entry = CacheEntry(
                key=key,
                prompt=full_prompt,
                response=response,
                model=model,
                embedding=embedding,
                expires_at=expires_at,
                metadata=metadata or {}
            )

            self.backend.set(entry)

    def _semantic_search(
        self,
        prompt: str,
        model: str
    ) -> Optional[str]:
        """Search for semantically similar cached prompts."""
        try:
            query_embedding = self.embedding_provider.embed(prompt)
        except Exception:
            return None

        best_match: Optional[CacheEntry] = None
        best_similarity = self.similarity_threshold

        for entry in self.backend.get_all_entries():
            # Skip if different model or no embedding
            if entry.model != model or entry.embedding is None:
                continue

            # Skip expired entries
            if entry.is_expired():
                continue

            similarity = cosine_similarity(query_embedding, entry.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match is not None:
            return best_match.response

        return None

    def _update_latency(self, latency_ms: float) -> None:
        """Update average latency statistic."""
        total_hits = self.stats.hits + self.stats.semantic_hits
        if total_hits == 1:
            self.stats.avg_hit_latency_ms = latency_ms
        else:
            # Running average
            self.stats.avg_hit_latency_ms = (
                (self.stats.avg_hit_latency_ms * (total_hits - 1) + latency_ms)
                / total_hits
            )

    def _update_cost_saved(self, response: str) -> None:
        """Update estimated cost saved."""
        tokens = self._estimate_tokens(response)
        cost = (tokens / 1000) * self.cost_per_1k_tokens
        self.stats.estimated_cost_saved += cost

    def invalidate(self, prompt: str, model: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            prompt: The prompt to invalidate
            model: The model name

        Returns:
            True if entry was found and deleted
        """
        key = self._generate_key(prompt, model)
        return self.backend.delete(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.backend.clear()
        self.stats = CacheStats()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.

        Returns:
            Number of entries removed
        """
        removed = 0
        for entry in self.backend.get_all_entries():
            if entry.is_expired():
                if self.backend.delete(entry.key):
                    removed += 1
                    self.stats.evictions += 1
        return removed


class CachedAIClient:
    """
    Wrapper that adds caching to any AI client.

    This wrapper intercepts calls to generate/chat methods and
    checks the cache before making API calls.

    Args:
        client: The underlying AI client (GeminiClient, ClaudeClient, etc.)
        cache: SemanticCache instance
        cache_chat: Whether to cache chat responses

    Example:
        >>> from ai_assistant import AIAssistant
        >>> from ai_assistant.cache import SemanticCache, CachedAIClient
        >>>
        >>> assistant = AIAssistant(gemini_api_key="...")
        >>> cache = SemanticCache()
        >>> cached_client = CachedAIClient(assistant.claude, cache)
        >>>
        >>> # First call hits API
        >>> response1 = cached_client.generate("What is Spark?")
        >>> # Second call returns cached response
        >>> response2 = cached_client.generate("What is Spark?")
    """

    def __init__(
        self,
        client: Any,
        cache: SemanticCache,
        cache_chat: bool = False
    ):
        self.client = client
        self.cache = cache
        self.cache_chat = cache_chat

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response with caching.

        Args:
            prompt: The input prompt
            system_instruction: Optional system instruction
            **kwargs: Additional arguments for the client

        Returns:
            Generated or cached response
        """
        model_name = getattr(
            self.client, 'model_config',
            type(self.client).__name__
        )
        if hasattr(model_name, 'name'):
            model_name = model_name.name

        # Try cache first
        cached = self.cache.get(prompt, str(model_name), system_instruction)
        if cached is not None:
            return cached

        # Call the actual client
        response = self.client.generate(
            prompt,
            system_instruction=system_instruction,
            **kwargs
        )

        # Store in cache
        self.cache.set(
            prompt,
            response,
            str(model_name),
            system_instruction
        )

        return response

    def chat(
        self,
        message: str,
        conversation_name: str = "default",
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Chat with caching (if enabled).

        Note: Chat caching is disabled by default as conversations
        are context-dependent.
        """
        if not self.cache_chat:
            return self.client.chat(
                message,
                conversation_name,
                system_instruction
            )

        # For chat caching, include conversation name in key
        cache_key = f"[CHAT:{conversation_name}]{message}"
        model_name = getattr(
            self.client, 'model_config',
            type(self.client).__name__
        )
        if hasattr(model_name, 'name'):
            model_name = model_name.name

        cached = self.cache.get(cache_key, str(model_name), system_instruction)
        if cached is not None:
            return cached

        response = self.client.chat(
            message,
            conversation_name,
            system_instruction
        )

        self.cache.set(
            cache_key,
            response,
            str(model_name),
            system_instruction
        )

        return response

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying client."""
        return getattr(self.client, name)


def create_cache(
    backend_type: str = "memory",
    max_size: int = 1000,
    ttl_seconds: int = 3600,
    similarity_threshold: float = 0.95,
    db_path: Optional[str] = None,
    embedding_api_key: Optional[str] = None
) -> SemanticCache:
    """
    Factory function to create a SemanticCache with common configurations.

    Args:
        backend_type: "memory" or "sqlite"
        max_size: Maximum cache size
        ttl_seconds: Time-to-live in seconds
        similarity_threshold: Semantic similarity threshold
        db_path: Path for SQLite database (required if backend_type="sqlite")
        embedding_api_key: API key for embedding provider

    Returns:
        Configured SemanticCache instance

    Example:
        >>> # Simple memory cache
        >>> cache = create_cache()
        >>>
        >>> # Persistent SQLite cache
        >>> cache = create_cache(
        ...     backend_type="sqlite",
        ...     db_path="/tmp/ai_cache.db",
        ...     ttl_seconds=86400  # 24 hours
        ... )
    """
    # Create backend
    if backend_type == "sqlite":
        if db_path is None:
            db_path = "/tmp/ai_assistant_cache.db"
        backend = SQLiteCacheBackend(db_path, max_size)
    else:
        backend = MemoryCacheBackend(max_size)

    # Create embedding provider
    if embedding_api_key:
        embedding_provider = GeminiEmbeddingProvider(embedding_api_key)
    else:
        embedding_provider = SimpleHashEmbedding()

    return SemanticCache(
        backend=backend,
        embedding_provider=embedding_provider,
        similarity_threshold=similarity_threshold,
        ttl_seconds=ttl_seconds
    )
