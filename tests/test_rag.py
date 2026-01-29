"""
Unit tests for RAG (Retrieval Augmented Generation) module.

Tests document chunking, vector stores, context providers,
and the RAG pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from ai_assistant.rag import (
    Document,
    Chunk,
    ChunkingStrategy,
    DocumentChunker,
    VectorStore,
    InMemoryVectorStore,
    UnityCatalogContextProvider,
    RAGPipeline,
    create_rag_pipeline
)


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="doc_1",
            content="Test content",
            metadata={"author": "Test"}
        )

        assert doc.content == "Test content"
        assert doc.id == "doc_1"
        assert doc.metadata == {"author": "Test"}

    def test_document_different_ids(self):
        """Test documents with different IDs."""
        doc1 = Document(id="doc_1", content="Content 1")
        doc2 = Document(id="doc_2", content="Content 2")

        assert doc1.id != doc2.id


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            id="chunk_1",
            content="Chunk content",
            document_id="doc_1",
            chunk_index=0,
            embedding=[0.1, 0.2]
        )

        assert chunk.content == "Chunk content"
        assert chunk.chunk_index == 0
        assert chunk.embedding == [0.1, 0.2]
        assert chunk.document_id == "doc_1"


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a chunker with default settings."""
        return DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=100,
            chunk_overlap=20
        )

    def test_chunker_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.strategy == ChunkingStrategy.FIXED_SIZE
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20

    def test_fixed_size_chunking(self, chunker):
        """Test fixed size chunking."""
        doc = Document(
            id="test_doc",
            content="A" * 250  # 250 characters
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=100
        )

        doc = Document(
            id="test_doc",
            content="First sentence. Second sentence. Third sentence here."
        )

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_paragraph_chunking(self):
        """Test paragraph-based chunking."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=200
        )

        doc = Document(
            id="test_doc",
            content="Paragraph one.\n\nParagraph two.\n\nParagraph three."
        )

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_empty_document(self, chunker):
        """Test chunking empty document."""
        doc = Document(id="empty_doc", content="")
        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_chunk_metadata_preserved(self, chunker):
        """Test that document metadata is preserved in chunks."""
        doc = Document(
            id="test_doc",
            content="A" * 150,
            metadata={"key": "value"}
        )

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata.get("key") == "value"


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider."""
        from ai_assistant.cache import SimpleHashEmbedding
        return SimpleHashEmbedding(dimension=128)

    @pytest.fixture
    def store(self, mock_embedding_provider):
        """Create a vector store."""
        return InMemoryVectorStore(embedding_provider=mock_embedding_provider)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        return [
            Chunk(
                id="chunk_1",
                content="Python programming",
                document_id="doc1",
                chunk_index=0,
                embedding=[0.1] * 128
            ),
            Chunk(
                id="chunk_2",
                content="Data engineering",
                document_id="doc2",
                chunk_index=0,
                embedding=[0.2] * 128
            )
        ]

    def test_store_initialization(self, store):
        """Test store initialization."""
        assert store.embedding_provider is not None
        assert len(store._chunks) == 0

    def test_add_chunks(self, store, sample_chunks):
        """Test adding chunks."""
        store.add(sample_chunks)

        assert len(store._chunks) == 2

    def test_search(self, store, sample_chunks):
        """Test searching for similar chunks."""
        store.add(sample_chunks)

        results = store.search("Python programming", k=2)

        assert len(results) <= 2
        assert all(isinstance(r, Chunk) for r in results)

    def test_search_empty_store(self, store):
        """Test searching empty store."""
        results = store.search("test query", k=5)

        assert results == []

    def test_search_top_k(self, store, sample_chunks):
        """Test top_k parameter."""
        store.add(sample_chunks)

        results = store.search("Python", k=1)
        assert len(results) == 1

    def test_clear_store(self, store, sample_chunks):
        """Test clearing the store."""
        store.add(sample_chunks)
        store.clear()

        assert len(store._chunks) == 0


class TestUnityCatalogContextProvider:
    """Tests for UnityCatalogContextProvider."""

    @pytest.fixture
    def mock_spark(self):
        """Create a mock Spark session."""
        spark = MagicMock()
        return spark

    @pytest.fixture
    def provider(self, mock_spark):
        """Create a context provider."""
        return UnityCatalogContextProvider(
            spark=mock_spark,
            catalog="test_catalog",
            schema="schema1"
        )

    def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.default_catalog == "test_catalog"
        assert provider.default_schema == "schema1"

    def test_get_table_context(self, provider, mock_spark):
        """Test getting context for tables."""
        # Mock the SQL execution for DESCRIBE TABLE
        mock_result = MagicMock()
        mock_result.collect.return_value = [
            MagicMock(__getitem__=lambda self, i: ["id", "int", "Primary key"][i]),
            MagicMock(__getitem__=lambda self, i: ["name", "string", "User name"][i]),
        ]
        mock_spark.sql.return_value = mock_result

        context = provider.get_table_context("table1", include_sample=False)

        assert context is not None
        assert isinstance(context, str)

    def test_get_schema_context(self, provider, mock_spark):
        """Test getting schema context."""
        # Mock SHOW TABLES result
        mock_result = MagicMock()
        mock_result.collect.return_value = [
            MagicMock(tableName="table1"),
            MagicMock(tableName="table2")
        ]
        mock_spark.sql.return_value = mock_result

        context = provider.get_schema_context()

        assert isinstance(context, str)


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.generate = Mock(return_value="Generated answer based on context")
        return client

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider."""
        from ai_assistant.cache import SimpleHashEmbedding
        return SimpleHashEmbedding(dimension=128)

    @pytest.fixture
    def pipeline(self, mock_ai_client, mock_embedding_provider):
        """Create a RAG pipeline."""
        vector_store = InMemoryVectorStore(embedding_provider=mock_embedding_provider)
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        return RAGPipeline(
            vector_store=vector_store,
            chunker=chunker,
            ai_client=mock_ai_client,
            default_k=3
        )

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.chunker.chunk_size == 100
        assert pipeline.default_k == 3

    def test_add_documents(self, pipeline):
        """Test adding documents."""
        docs = [
            Document(id="doc1", content="Document 1 content"),
            Document(id="doc2", content="Document 2 content")
        ]

        pipeline.add_documents(docs)

        # Should have chunked and added to store
        assert len(pipeline.vector_store._chunks) > 0

    def test_query(self, pipeline, mock_ai_client):
        """Test querying the pipeline."""
        # Add a document first
        docs = [Document(id="test_doc", content="Python is a programming language.")]
        pipeline.add_documents(docs)

        response = pipeline.query("What is Python?")

        assert response is not None
        mock_ai_client.generate.assert_called_once()

    def test_retrieve(self, pipeline):
        """Test retrieving documents."""
        docs = [Document(id="test_doc", content="Test content about Python programming")]
        pipeline.add_documents(docs)

        result = pipeline.retrieve("Python")

        assert result is not None
        assert isinstance(result.chunks, list)

    def test_clear_pipeline(self, pipeline):
        """Test clearing the pipeline."""
        docs = [Document(id="test_doc", content="Test")]
        pipeline.add_documents(docs)

        pipeline.clear()

        assert len(pipeline.vector_store._chunks) == 0


class TestCreateRAGPipeline:
    """Tests for factory function."""

    def test_create_rag_pipeline(self):
        """Test creating a RAG pipeline."""
        pipeline = create_rag_pipeline(
            chunk_size=200,
            chunk_overlap=50
        )

        assert isinstance(pipeline, RAGPipeline)
        assert pipeline.chunker.chunk_size == 200
        assert pipeline.chunker.chunk_overlap == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
