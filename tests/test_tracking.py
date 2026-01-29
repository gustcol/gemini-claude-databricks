"""
Unit tests for MLflow Tracking module.

Tests the AI tracking system including metrics, experiments,
and the tracked AI client wrapper.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from ai_assistant.tracking import (
    LLMCallMetrics,
    AITracker,
    TrackedAIClient,
    ABExperiment,
    create_tracker
)


class TestLLMCallMetrics:
    """Tests for LLMCallMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating metrics."""
        metrics = LLMCallMetrics(
            model="gemini-1.5-pro",
            prompt="Test prompt",
            response="Test response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=150.5,
            cost=0.001
        )

        assert metrics.model == "gemini-1.5-pro"
        assert metrics.input_tokens == 10
        assert metrics.output_tokens == 20
        assert metrics.latency_ms == 150.5
        assert metrics.cost == 0.001
        assert metrics.timestamp is not None

    def test_metrics_defaults(self):
        """Test default values."""
        metrics = LLMCallMetrics(
            model="claude",
            prompt="Test",
            response="Response",
            input_tokens=5,
            output_tokens=10,
            latency_ms=100
        )

        assert metrics.cost == 0.0
        assert metrics.metadata == {}

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = LLMCallMetrics(
            model="gemini",
            prompt="Test",
            response="Response",
            input_tokens=5,
            output_tokens=10,
            latency_ms=100
        )

        data = metrics.to_dict()

        assert data["model"] == "gemini"
        assert data["input_tokens"] == 5
        assert "timestamp" in data


class TestAITracker:
    """Tests for AITracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker without MLflow."""
        return AITracker(
            experiment_name="test_experiment",
            enable_mlflow=False
        )

    @pytest.fixture
    def mock_mlflow(self):
        """Create mock MLflow module."""
        with patch('ai_assistant.tracking.mlflow') as mock:
            mock.start_run = MagicMock()
            mock.end_run = MagicMock()
            mock.log_metric = MagicMock()
            mock.log_param = MagicMock()
            mock.set_experiment = MagicMock()
            yield mock

    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.experiment_name == "test_experiment"
        assert tracker.enable_mlflow is False

    def test_log_call(self, tracker):
        """Test logging an LLM call."""
        metrics = LLMCallMetrics(
            model="gemini",
            prompt="Hello",
            response="Hi",
            input_tokens=2,
            output_tokens=1,
            latency_ms=50
        )

        tracker.log_call(metrics)

        assert len(tracker._calls) == 1
        assert tracker._calls[0] == metrics

    def test_get_stats(self, tracker):
        """Test getting statistics."""
        metrics1 = LLMCallMetrics(
            model="gemini",
            prompt="Test1",
            response="Response1",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100,
            cost=0.01
        )
        metrics2 = LLMCallMetrics(
            model="gemini",
            prompt="Test2",
            response="Response2",
            input_tokens=15,
            output_tokens=25,
            latency_ms=150,
            cost=0.02
        )

        tracker.log_call(metrics1)
        tracker.log_call(metrics2)

        stats = tracker.get_stats()

        assert stats["total_calls"] == 2
        assert stats["total_input_tokens"] == 25
        assert stats["total_output_tokens"] == 45
        assert stats["total_cost"] == pytest.approx(0.03)
        assert stats["avg_latency_ms"] == pytest.approx(125.0)

    def test_get_stats_empty(self, tracker):
        """Test getting stats with no calls."""
        stats = tracker.get_stats()

        assert stats["total_calls"] == 0

    def test_get_stats_by_model(self, tracker):
        """Test getting stats by model."""
        metrics1 = LLMCallMetrics(
            model="gemini",
            prompt="Test",
            response="Response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100
        )
        metrics2 = LLMCallMetrics(
            model="claude",
            prompt="Test",
            response="Response",
            input_tokens=15,
            output_tokens=25,
            latency_ms=150
        )

        tracker.log_call(metrics1)
        tracker.log_call(metrics2)

        stats = tracker.get_stats(model="gemini")

        assert stats["total_calls"] == 1
        assert stats["total_input_tokens"] == 10

    def test_clear_history(self, tracker):
        """Test clearing call history."""
        metrics = LLMCallMetrics(
            model="gemini",
            prompt="Test",
            response="Response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100
        )

        tracker.log_call(metrics)
        tracker.clear()

        assert len(tracker._calls) == 0

    def test_export_calls(self, tracker):
        """Test exporting calls."""
        metrics = LLMCallMetrics(
            model="gemini",
            prompt="Test",
            response="Response",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100
        )

        tracker.log_call(metrics)
        exported = tracker.export_calls()

        assert len(exported) == 1
        assert exported[0]["model"] == "gemini"


class TestTrackedAIClient:
    """Tests for TrackedAIClient wrapper."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.generate = Mock(return_value="Generated response")
        client.model_name = "test-model"
        return client

    @pytest.fixture
    def tracker(self):
        """Create a tracker."""
        return AITracker(enable_mlflow=False)

    @pytest.fixture
    def tracked_client(self, mock_client, tracker):
        """Create a tracked client."""
        return TrackedAIClient(
            client=mock_client,
            tracker=tracker,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002
        )

    def test_tracked_client_initialization(self, tracked_client):
        """Test tracked client initialization."""
        assert tracked_client.cost_per_1k_input == 0.001
        assert tracked_client.cost_per_1k_output == 0.002

    def test_generate_tracks_call(self, tracked_client, tracker, mock_client):
        """Test that generate tracks the call."""
        response = tracked_client.generate("Test prompt")

        assert response == "Generated response"
        assert len(tracker._calls) == 1
        mock_client.generate.assert_called_once()

    def test_generate_calculates_metrics(self, tracked_client, tracker):
        """Test that metrics are calculated."""
        tracked_client.generate("Test prompt with some words")

        call = tracker._calls[0]
        assert call.input_tokens > 0
        assert call.latency_ms >= 0

    def test_get_tracking_stats(self, tracked_client):
        """Test getting tracking stats."""
        tracked_client.generate("Test")

        stats = tracked_client.get_tracking_stats()

        assert stats["total_calls"] == 1


class TestABExperiment:
    """Tests for ABExperiment class."""

    @pytest.fixture
    def mock_client_a(self):
        """Create mock client A."""
        client = Mock()
        client.generate = Mock(return_value="Response from A")
        client.model_name = "model-a"
        return client

    @pytest.fixture
    def mock_client_b(self):
        """Create mock client B."""
        client = Mock()
        client.generate = Mock(return_value="Response from B")
        client.model_name = "model-b"
        return client

    @pytest.fixture
    def experiment(self, mock_client_a, mock_client_b):
        """Create an A/B experiment."""
        return ABExperiment(
            name="test_ab",
            client_a=mock_client_a,
            client_b=mock_client_b,
            split_ratio=0.5
        )

    def test_experiment_initialization(self, experiment):
        """Test experiment initialization."""
        assert experiment.name == "test_ab"
        assert experiment.split_ratio == 0.5

    def test_experiment_run(self, experiment):
        """Test running experiment."""
        # Run multiple times to test split
        responses = [experiment.run("Test") for _ in range(10)]

        # Should have some responses from each client
        assert len(responses) == 10
        assert all(r is not None for r in responses)

    def test_experiment_results(self, experiment):
        """Test getting experiment results."""
        # Run some calls
        for _ in range(10):
            experiment.run("Test")

        results = experiment.get_results()

        assert "model_a_calls" in results
        assert "model_b_calls" in results
        assert results["total_calls"] == 10


class TestCreateTracker:
    """Tests for factory function."""

    def test_create_tracker(self):
        """Test creating a tracker."""
        tracker = create_tracker(
            experiment_name="my_experiment",
            enable_mlflow=False
        )

        assert isinstance(tracker, AITracker)
        assert tracker.experiment_name == "my_experiment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
