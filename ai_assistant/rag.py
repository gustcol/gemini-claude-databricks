"""
RAG (Retrieval Augmented Generation) Module for AI Assistant.

This module provides RAG capabilities for enriching LLM prompts with
context from Unity Catalog tables, documents, and vector stores.

Features:
- Unity Catalog metadata retrieval
- Document chunking and embedding
- Vector similarity search
- Context assembly for prompts
- Integration with Databricks Vector Search
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
from enum import Enum
import re


class ChunkingStrategy(Enum):
    """Strategies for chunking documents."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


@dataclass
class Document:
    """
    Represents a document for RAG.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Additional metadata (source, type, etc.)
        embedding: Optional pre-computed embedding
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding
        }


@dataclass
class Chunk:
    """
    Represents a chunk of a document.

    Attributes:
        id: Unique chunk identifier
        content: Chunk text content
        document_id: Parent document ID
        chunk_index: Index within the document
        metadata: Additional metadata
        embedding: Optional pre-computed embedding
        score: Relevance score (set during retrieval)
    """
    id: str
    content: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "score": self.score
        }


@dataclass
class RetrievalResult:
    """
    Result of a retrieval operation.

    Attributes:
        chunks: Retrieved chunks sorted by relevance
        query: Original query
        total_results: Total number of results found
        metadata: Additional result metadata
    """
    chunks: List[Chunk]
    query: str
    total_results: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """Get combined context from all chunks."""
        return "\n\n".join([chunk.content for chunk in self.chunks])

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "query": self.query,
            "total_results": self.total_results,
            "context": self.context,
            "metadata": self.metadata
        }


class DocumentChunker:
    """
    Splits documents into chunks for embedding and retrieval.

    Args:
        strategy: Chunking strategy to use
        chunk_size: Target chunk size (in characters)
        chunk_overlap: Overlap between chunks (in characters)
        separator: Custom separator for splitting

    Example:
        >>> chunker = DocumentChunker(
        ...     strategy=ChunkingStrategy.PARAGRAPH,
        ...     chunk_size=1000,
        ...     chunk_overlap=100
        ... )
        >>> chunks = chunker.chunk(document)
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separator: Optional[str] = None
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(document)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentences(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraphs(document)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(document)
        else:
            return self._chunk_fixed_size(document)

    def _chunk_fixed_size(self, document: Document) -> List[Chunk]:
        """Chunk by fixed character size with overlap."""
        chunks = []
        text = document.content
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a word boundary
            if end < len(text):
                # Look for space or newline near the end
                for i in range(min(50, end - start)):
                    if text[end - i] in ' \n':
                        end = end - i
                        break

            chunk_content = text[start:end].strip()

            if chunk_content:
                chunk_id = f"{document.id}_chunk_{chunk_index}"
                chunks.append(Chunk(
                    id=chunk_id,
                    content=chunk_content,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata={
                        **document.metadata,
                        "start_char": start,
                        "end_char": end
                    }
                ))
                chunk_index += 1

            start = end - self.chunk_overlap

        return chunks

    def _chunk_sentences(self, document: Document) -> List[Chunk]:
        """Chunk by sentences, combining up to chunk_size."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', document.content)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = ' '.join(current_chunk)
                chunk_id = f"{document.id}_chunk_{chunk_index}"
                chunks.append(Chunk(
                    id=chunk_id,
                    content=chunk_content,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata=document.metadata
                ))
                chunk_index += 1

                # Start new chunk with overlap
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk_id = f"{document.id}_chunk_{chunk_index}"
            chunks.append(Chunk(
                id=chunk_id,
                content=chunk_content,
                document_id=document.id,
                chunk_index=chunk_index,
                metadata=document.metadata
            ))

        return chunks

    def _chunk_paragraphs(self, document: Document) -> List[Chunk]:
        """Chunk by paragraphs, combining short paragraphs."""
        paragraphs = document.content.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = '\n\n'.join(current_chunk)
                chunk_id = f"{document.id}_chunk_{chunk_index}"
                chunks.append(Chunk(
                    id=chunk_id,
                    content=chunk_content,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    metadata=document.metadata
                ))
                chunk_index += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunk_id = f"{document.id}_chunk_{chunk_index}"
            chunks.append(Chunk(
                id=chunk_id,
                content=chunk_content,
                document_id=document.id,
                chunk_index=chunk_index,
                metadata=document.metadata
            ))

        return chunks

    def _chunk_semantic(self, document: Document) -> List[Chunk]:
        """
        Semantic chunking based on content structure.

        Falls back to paragraph chunking for now.
        Could be enhanced with LLM-based chunking.
        """
        return self._chunk_paragraphs(document)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete chunks by ID."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all chunks."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for development and testing.

    Args:
        embedding_provider: Provider for generating embeddings

    Example:
        >>> from ai_assistant.cache import SimpleHashEmbedding
        >>> store = InMemoryVectorStore(SimpleHashEmbedding())
        >>> store.add(chunks)
        >>> results = store.search("query", k=5)
    """

    def __init__(self, embedding_provider: Any):
        self.embedding_provider = embedding_provider
        self._chunks: Dict[str, Chunk] = {}

    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks with embeddings to the store."""
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = self.embedding_provider.embed(chunk.content)
            self._chunks[chunk.id] = chunk

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Search for similar chunks using cosine similarity."""
        from .cache import cosine_similarity

        query_embedding = self.embedding_provider.embed(query)

        scored_chunks = []
        for chunk in self._chunks.values():
            # Apply metadata filter
            if filter_metadata:
                match = all(
                    chunk.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue

            if chunk.embedding:
                score = cosine_similarity(query_embedding, chunk.embedding)
                chunk_copy = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                    embedding=chunk.embedding,
                    score=score
                )
                scored_chunks.append(chunk_copy)

        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:k]

    def delete(self, ids: List[str]) -> None:
        """Delete chunks by ID."""
        for chunk_id in ids:
            self._chunks.pop(chunk_id, None)

    def clear(self) -> None:
        """Clear all chunks."""
        self._chunks.clear()

    def size(self) -> int:
        """Get number of chunks in store."""
        return len(self._chunks)


class DatabricksVectorSearchStore(VectorStore):
    """
    Databricks Vector Search integration.

    This store uses Databricks Vector Search for scalable
    vector similarity search.

    Args:
        endpoint_name: Vector Search endpoint name
        index_name: Vector Search index name
        embedding_column: Column name for embeddings
        text_column: Column name for text content
        spark: SparkSession instance

    Example:
        >>> store = DatabricksVectorSearchStore(
        ...     endpoint_name="vs-endpoint",
        ...     index_name="my_catalog.my_schema.my_index",
        ...     spark=spark
        ... )
    """

    def __init__(
        self,
        endpoint_name: str,
        index_name: str,
        embedding_column: str = "embedding",
        text_column: str = "content",
        spark: Any = None
    ):
        self.endpoint_name = endpoint_name
        self.index_name = index_name
        self.embedding_column = embedding_column
        self.text_column = text_column
        self.spark = spark
        self._client = None

    def _get_client(self):
        """Get or create Vector Search client."""
        if self._client is None:
            try:
                from databricks.vector_search.client import VectorSearchClient
                self._client = VectorSearchClient()
            except ImportError:
                raise ImportError(
                    "databricks-vector-search package not installed. "
                    "Install with: pip install databricks-vector-search"
                )
        return self._client

    def add(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to Vector Search.

        Note: In Databricks, this typically means writing to the
        source Delta table that the index is built on.
        """
        if self.spark is None:
            raise ValueError("SparkSession required for adding chunks")

        # Convert chunks to rows
        rows = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "metadata": json.dumps(chunk.metadata)
            }
            for chunk in chunks
        ]

        # Create DataFrame and write to source table
        df = self.spark.createDataFrame(rows)

        # Extract table name from index name (catalog.schema.table)
        source_table = self.index_name.rsplit("_index", 1)[0]

        df.write.mode("append").saveAsTable(source_table)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Search using Databricks Vector Search."""
        client = self._get_client()
        index = client.get_index(
            endpoint_name=self.endpoint_name,
            index_name=self.index_name
        )

        # Build filters
        filters = None
        if filter_metadata:
            filter_conditions = [
                f"{key} = '{value}'"
                for key, value in filter_metadata.items()
            ]
            filters = " AND ".join(filter_conditions)

        # Execute search
        results = index.similarity_search(
            query_text=query,
            columns=[self.text_column, "document_id", "chunk_index", "metadata"],
            num_results=k,
            filters=filters
        )

        # Convert to Chunk objects
        chunks = []
        for row in results.get("result", {}).get("data_array", []):
            chunk = Chunk(
                id=row[0],
                content=row[1],
                document_id=row[2],
                chunk_index=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
                score=row[-1]  # Score is typically last
            )
            chunks.append(chunk)

        return chunks

    def delete(self, ids: List[str]) -> None:
        """Delete chunks from source table."""
        if self.spark is None:
            raise ValueError("SparkSession required for deleting chunks")

        source_table = self.index_name.rsplit("_index", 1)[0]
        id_list = ", ".join([f"'{id}'" for id in ids])
        self.spark.sql(f"DELETE FROM {source_table} WHERE id IN ({id_list})")

    def clear(self) -> None:
        """Clear all chunks from source table."""
        if self.spark is None:
            raise ValueError("SparkSession required for clearing chunks")

        source_table = self.index_name.rsplit("_index", 1)[0]
        self.spark.sql(f"TRUNCATE TABLE {source_table}")


class UnityCatalogContextProvider:
    """
    Provides context from Unity Catalog metadata.

    This class retrieves table schemas, comments, and statistics
    to enrich prompts with database context.

    Args:
        spark: SparkSession instance
        catalog: Default catalog name
        schema: Default schema name

    Example:
        >>> provider = UnityCatalogContextProvider(spark, "my_catalog")
        >>> context = provider.get_table_context("my_schema.my_table")
    """

    def __init__(
        self,
        spark: Any,
        catalog: Optional[str] = None,
        schema: Optional[str] = None
    ):
        self.spark = spark
        self.default_catalog = catalog
        self.default_schema = schema

    def get_table_context(
        self,
        table_name: str,
        include_sample: bool = True,
        sample_rows: int = 5
    ) -> str:
        """
        Get context information for a table.

        Args:
            table_name: Table name (can include catalog.schema prefix)
            include_sample: Whether to include sample data
            sample_rows: Number of sample rows to include

        Returns:
            Formatted context string
        """
        # Parse table name
        full_name = self._resolve_table_name(table_name)

        context_parts = []

        # Get table description
        try:
            desc = self.spark.sql(f"DESCRIBE TABLE EXTENDED {full_name}")
            desc_rows = desc.collect()

            # Extract schema
            schema_lines = []
            in_schema = True
            table_comment = ""

            for row in desc_rows:
                col_name = row[0]
                if col_name.strip() == "":
                    in_schema = False
                    continue

                if in_schema and col_name and not col_name.startswith("#"):
                    col_type = row[1] if row[1] else ""
                    col_comment = row[2] if len(row) > 2 and row[2] else ""
                    schema_lines.append(
                        f"  - {col_name}: {col_type}"
                        + (f" -- {col_comment}" if col_comment else "")
                    )

                if col_name == "Comment":
                    table_comment = row[1] if row[1] else ""

            context_parts.append(f"## Table: {full_name}")
            if table_comment:
                context_parts.append(f"Description: {table_comment}")
            context_parts.append("\n### Schema:")
            context_parts.extend(schema_lines)

        except Exception as e:
            context_parts.append(f"## Table: {full_name}")
            context_parts.append(f"Error retrieving schema: {str(e)}")

        # Get sample data
        if include_sample:
            try:
                sample_df = self.spark.sql(
                    f"SELECT * FROM {full_name} LIMIT {sample_rows}"
                )
                sample_str = sample_df.toPandas().to_string()
                context_parts.append("\n### Sample Data:")
                context_parts.append(f"```\n{sample_str}\n```")
            except Exception as e:
                context_parts.append(f"\nCould not retrieve sample: {str(e)}")

        return "\n".join(context_parts)

    def get_schema_context(self, schema_name: Optional[str] = None) -> str:
        """
        Get context for all tables in a schema.

        Args:
            schema_name: Schema name (uses default if not provided)

        Returns:
            Formatted context string
        """
        schema = schema_name or self.default_schema
        catalog = self.default_catalog

        if not schema:
            return "No schema specified"

        full_schema = f"{catalog}.{schema}" if catalog else schema

        try:
            tables = self.spark.sql(f"SHOW TABLES IN {full_schema}")
            table_rows = tables.collect()

            context_parts = [f"## Schema: {full_schema}\n"]
            context_parts.append("### Tables:")

            for row in table_rows:
                table_name = row.tableName
                context_parts.append(f"\n#### {table_name}")

                # Get brief table info
                try:
                    desc = self.spark.sql(
                        f"DESCRIBE TABLE {full_schema}.{table_name}"
                    )
                    columns = [
                        f"{r[0]}: {r[1]}"
                        for r in desc.collect()
                        if r[0] and not r[0].startswith("#")
                    ][:5]  # First 5 columns
                    context_parts.append(f"Columns: {', '.join(columns)}")
                    if len(columns) == 5:
                        context_parts.append("...")
                except Exception:
                    pass

            return "\n".join(context_parts)

        except Exception as e:
            return f"Error retrieving schema info: {str(e)}"

    def search_tables(
        self,
        pattern: str,
        catalog: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Search for tables matching a pattern.

        Args:
            pattern: Search pattern (supports SQL LIKE syntax)
            catalog: Catalog to search in

        Returns:
            List of matching table information
        """
        cat = catalog or self.default_catalog

        try:
            if cat:
                query = f"""
                    SELECT table_catalog, table_schema, table_name, comment
                    FROM {cat}.information_schema.tables
                    WHERE table_name LIKE '%{pattern}%'
                    OR comment LIKE '%{pattern}%'
                    LIMIT 20
                """
            else:
                query = f"SHOW TABLES LIKE '*{pattern}*'"

            results = self.spark.sql(query).collect()

            tables = []
            for row in results:
                if hasattr(row, 'table_catalog'):
                    tables.append({
                        "catalog": row.table_catalog,
                        "schema": row.table_schema,
                        "table": row.table_name,
                        "comment": row.comment or ""
                    })
                else:
                    tables.append({
                        "table": row.tableName,
                        "database": row.database
                    })

            return tables

        except Exception as e:
            return [{"error": str(e)}]

    def _resolve_table_name(self, table_name: str) -> str:
        """Resolve a table name to full catalog.schema.table format."""
        parts = table_name.split(".")

        if len(parts) == 3:
            return table_name
        elif len(parts) == 2:
            if self.default_catalog:
                return f"{self.default_catalog}.{table_name}"
            return table_name
        else:
            if self.default_catalog and self.default_schema:
                return f"{self.default_catalog}.{self.default_schema}.{table_name}"
            elif self.default_schema:
                return f"{self.default_schema}.{table_name}"
            return table_name


class RAGPipeline:
    """
    Complete RAG pipeline for AI Assistant.

    This class orchestrates document processing, vector storage,
    retrieval, and context assembly for RAG-enhanced LLM calls.

    Args:
        vector_store: Vector store for similarity search
        chunker: Document chunker
        uc_context_provider: Unity Catalog context provider
        ai_client: AI client for generation
        default_k: Default number of chunks to retrieve
        context_template: Template for assembling context

    Example:
        >>> from ai_assistant.cache import SimpleHashEmbedding
        >>>
        >>> # Create components
        >>> store = InMemoryVectorStore(SimpleHashEmbedding())
        >>> chunker = DocumentChunker(chunk_size=500)
        >>> rag = RAGPipeline(vector_store=store, chunker=chunker)
        >>>
        >>> # Add documents
        >>> rag.add_documents([doc1, doc2])
        >>>
        >>> # Query with RAG
        >>> response = rag.query("What is Delta Lake?", ai_client)
    """

    DEFAULT_CONTEXT_TEMPLATE = """Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

## Context:
{context}

## Question:
{question}

## Answer:"""

    def __init__(
        self,
        vector_store: VectorStore,
        chunker: Optional[DocumentChunker] = None,
        uc_context_provider: Optional[UnityCatalogContextProvider] = None,
        ai_client: Any = None,
        default_k: int = 5,
        context_template: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.chunker = chunker or DocumentChunker()
        self.uc_context_provider = uc_context_provider
        self.ai_client = ai_client
        self.default_k = default_k
        self.context_template = context_template or self.DEFAULT_CONTEXT_TEMPLATE

    def add_documents(self, documents: List[Document]) -> int:
        """
        Process and add documents to the vector store.

        Args:
            documents: Documents to add

        Returns:
            Number of chunks created
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        self.vector_store.add(all_chunks)
        return len(all_chunks)

    def add_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add text content as a document.

        Args:
            text: Text content
            document_id: Optional document ID
            metadata: Optional metadata

        Returns:
            Number of chunks created
        """
        doc_id = document_id or hashlib.md5(text.encode()).hexdigest()
        doc = Document(
            id=doc_id,
            content=text,
            metadata=metadata or {}
        )
        return self.add_documents([doc])

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_uc_context: bool = False,
        uc_tables: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query
            k: Number of chunks to retrieve
            filter_metadata: Metadata filters
            include_uc_context: Whether to include Unity Catalog context
            uc_tables: Specific tables to include context from

        Returns:
            RetrievalResult with relevant chunks
        """
        k = k or self.default_k

        # Retrieve from vector store
        chunks = self.vector_store.search(query, k, filter_metadata)

        # Add Unity Catalog context if requested
        metadata = {}
        if include_uc_context and self.uc_context_provider:
            uc_context_parts = []

            if uc_tables:
                for table in uc_tables:
                    ctx = self.uc_context_provider.get_table_context(table)
                    uc_context_parts.append(ctx)
            else:
                # Try to detect table references in query
                tables = self.uc_context_provider.search_tables(query)
                for table_info in tables[:3]:  # Limit to 3 tables
                    if "table" in table_info:
                        table_name = table_info.get("table", "")
                        if table_info.get("catalog"):
                            full_name = f"{table_info['catalog']}.{table_info.get('schema', 'default')}.{table_name}"
                        else:
                            full_name = table_name
                        if full_name:
                            ctx = self.uc_context_provider.get_table_context(full_name)
                            uc_context_parts.append(ctx)

            if uc_context_parts:
                metadata["uc_context"] = "\n\n".join(uc_context_parts)

        return RetrievalResult(
            chunks=chunks,
            query=query,
            total_results=len(chunks),
            metadata=metadata
        )

    def query(
        self,
        question: str,
        ai_client: Any = None,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_uc_context: bool = False,
        uc_tables: Optional[List[str]] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Query with RAG-enhanced context.

        Args:
            question: User question
            ai_client: AI client (uses default if not provided)
            k: Number of chunks to retrieve
            filter_metadata: Metadata filters
            include_uc_context: Whether to include UC context
            uc_tables: Specific tables for UC context
            system_instruction: Optional system instruction

        Returns:
            AI-generated response
        """
        client = ai_client or self.ai_client
        if client is None:
            raise ValueError("No AI client provided")

        # Retrieve context
        result = self.retrieve(
            question,
            k=k,
            filter_metadata=filter_metadata,
            include_uc_context=include_uc_context,
            uc_tables=uc_tables
        )

        # Assemble context
        context_parts = [result.context]

        if result.metadata.get("uc_context"):
            context_parts.append("\n## Database Context:\n")
            context_parts.append(result.metadata["uc_context"])

        full_context = "\n".join(context_parts)

        # Build prompt
        prompt = self.context_template.format(
            context=full_context,
            question=question
        )

        # Generate response
        return client.generate(prompt, system_instruction=system_instruction)

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.clear()


def create_rag_pipeline(
    embedding_api_key: Optional[str] = None,
    spark: Any = None,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    vector_search_endpoint: Optional[str] = None,
    vector_search_index: Optional[str] = None
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.

    Args:
        embedding_api_key: API key for embeddings
        spark: SparkSession for Unity Catalog access
        catalog: Default catalog name
        schema: Default schema name
        chunk_size: Chunk size for documents
        chunk_overlap: Overlap between chunks
        vector_search_endpoint: Databricks Vector Search endpoint
        vector_search_index: Databricks Vector Search index

    Returns:
        Configured RAGPipeline instance

    Example:
        >>> # Simple in-memory RAG
        >>> rag = create_rag_pipeline()
        >>>
        >>> # With Databricks Vector Search
        >>> rag = create_rag_pipeline(
        ...     spark=spark,
        ...     catalog="my_catalog",
        ...     vector_search_endpoint="vs-endpoint",
        ...     vector_search_index="my_catalog.my_schema.docs_index"
        ... )
    """
    from .cache import SimpleHashEmbedding, GeminiEmbeddingProvider

    # Create embedding provider
    if embedding_api_key:
        embedding_provider = GeminiEmbeddingProvider(embedding_api_key)
    else:
        embedding_provider = SimpleHashEmbedding()

    # Create vector store
    if vector_search_endpoint and vector_search_index:
        vector_store = DatabricksVectorSearchStore(
            endpoint_name=vector_search_endpoint,
            index_name=vector_search_index,
            spark=spark
        )
    else:
        vector_store = InMemoryVectorStore(embedding_provider)

    # Create chunker
    chunker = DocumentChunker(
        strategy=ChunkingStrategy.PARAGRAPH,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Create UC context provider
    uc_provider = None
    if spark:
        uc_provider = UnityCatalogContextProvider(spark, catalog, schema)

    return RAGPipeline(
        vector_store=vector_store,
        chunker=chunker,
        uc_context_provider=uc_provider
    )
