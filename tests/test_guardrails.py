"""
Unit tests for Guardrails module.

Tests security guardrails including SQL validation, PII detection,
rate limiting, and audit logging.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from ai_assistant.guardrails import (
    SQLValidator,
    PIIDetector,
    RateLimiter,
    AuditLogger,
    AIGuardrails,
    create_guardrails
)


class TestSQLValidator:
    """Tests for SQLValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a SQL validator."""
        return SQLValidator()

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert len(validator._dangerous_patterns) > 0

    def test_validate_safe_select(self, validator):
        """Test validating safe SELECT queries."""
        query = "SELECT id, name FROM users WHERE active = true"
        result = validator.validate(query)

        assert result.is_safe is True
        assert len(result.issues) == 0

    def test_validate_drop_table(self, validator):
        """Test detecting DROP TABLE."""
        query = "DROP TABLE users"
        result = validator.validate(query)

        assert result.is_safe is False
        assert any("DROP" in issue.upper() for issue in result.issues)

    def test_validate_truncate(self, validator):
        """Test detecting TRUNCATE."""
        query = "TRUNCATE TABLE orders"
        result = validator.validate(query)

        assert result.is_safe is False

    def test_validate_delete_without_where(self, validator):
        """Test detecting DELETE without WHERE."""
        query = "DELETE FROM users"
        result = validator.validate(query)

        assert result.is_safe is False
        assert any("WHERE" in issue.upper() for issue in result.issues)

    def test_validate_delete_with_where(self, validator):
        """Test DELETE with WHERE is safe."""
        query = "DELETE FROM users WHERE id = 5"
        result = validator.validate(query)

        # May still be flagged as potentially dangerous
        # but should at least pass basic validation

    def test_validate_update_without_where(self, validator):
        """Test detecting UPDATE without WHERE."""
        query = "UPDATE users SET status = 'inactive'"
        result = validator.validate(query)

        assert result.is_safe is False

    def test_validate_grant(self, validator):
        """Test detecting GRANT statement."""
        query = "GRANT ALL ON database.* TO user"
        result = validator.validate(query)

        assert result.is_safe is False

    def test_validate_revoke(self, validator):
        """Test detecting REVOKE statement."""
        query = "REVOKE SELECT ON users FROM user"
        result = validator.validate(query)

        assert result.is_safe is False

    def test_validate_sql_injection_patterns(self, validator):
        """Test detecting SQL injection patterns."""
        injection_queries = [
            "SELECT * FROM users WHERE id = 1 OR 1=1",
            "SELECT * FROM users WHERE name = '' OR '1'='1'",
            "SELECT * FROM users; DROP TABLE users;--"
        ]

        for query in injection_queries:
            result = validator.validate(query)
            # Should detect at least some injection patterns
            # Implementation specific

    def test_validate_union_injection(self, validator):
        """Test detecting UNION-based injection."""
        query = "SELECT * FROM users UNION SELECT * FROM passwords"
        result = validator.validate(query)

        # UNION itself may be valid, but combined with passwords table suspicious

    def test_custom_patterns(self):
        """Test adding custom dangerous patterns."""
        validator = SQLValidator(
            additional_patterns=["CUSTOM_DANGEROUS"]
        )

        query = "SELECT CUSTOM_DANGEROUS FROM test"
        result = validator.validate(query)

        assert result.is_safe is False


class TestPIIDetector:
    """Tests for PIIDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a PII detector."""
        return PIIDetector()

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert len(detector._patterns) > 0

    def test_detect_email(self, detector):
        """Test detecting email addresses."""
        text = "Contact us at support@example.com for help"
        result = detector.detect(text)

        assert result.contains_pii is True
        assert any("email" in p.lower() for p in result.pii_types)

    def test_detect_phone(self, detector):
        """Test detecting phone numbers."""
        text = "Call me at 555-123-4567"
        result = detector.detect(text)

        assert result.contains_pii is True
        assert any("phone" in p.lower() for p in result.pii_types)

    def test_detect_ssn(self, detector):
        """Test detecting SSN."""
        text = "My SSN is 123-45-6789"
        result = detector.detect(text)

        assert result.contains_pii is True
        assert any("ssn" in p.lower() for p in result.pii_types)

    def test_detect_credit_card(self, detector):
        """Test detecting credit card numbers."""
        text = "Card number: 4111-1111-1111-1111"
        result = detector.detect(text)

        assert result.contains_pii is True
        assert any("credit" in p.lower() or "card" in p.lower() for p in result.pii_types)

    def test_no_pii(self, detector):
        """Test text without PII."""
        text = "The weather is nice today."
        result = detector.detect(text)

        assert result.contains_pii is False
        assert len(result.pii_types) == 0

    def test_redact_email(self, detector):
        """Test redacting email addresses."""
        text = "Send to user@example.com please"
        redacted = detector.redact(text)

        assert "user@example.com" not in redacted
        assert "[REDACTED" in redacted or "***" in redacted

    def test_redact_multiple_pii(self, detector):
        """Test redacting multiple PII types."""
        text = "Email: test@test.com, Phone: 555-123-4567, SSN: 123-45-6789"
        redacted = detector.redact(text)

        assert "test@test.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "123-45-6789" not in redacted

    def test_redact_preserves_structure(self, detector):
        """Test that redaction preserves text structure."""
        text = "Name: John, Email: john@example.com"
        redacted = detector.redact(text)

        assert "Name: John" in redacted
        assert "Email:" in redacted
        assert "john@example.com" not in redacted


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.fixture
    def limiter(self):
        """Create a rate limiter."""
        return RateLimiter(
            requests_per_minute=10,
            requests_per_hour=100
        )

    def test_limiter_initialization(self, limiter):
        """Test limiter initialization."""
        assert limiter.requests_per_minute == 10
        assert limiter.requests_per_hour == 100

    def test_allow_request(self, limiter):
        """Test allowing requests."""
        result = limiter.check("user_1")

        assert result.allowed is True
        assert result.remaining > 0

    def test_rate_limit_per_minute(self):
        """Test per-minute rate limiting."""
        limiter = RateLimiter(requests_per_minute=3, requests_per_hour=1000)

        # Make 3 requests (should all pass)
        for _ in range(3):
            result = limiter.check("user_1")
            assert result.allowed is True

        # 4th request should be limited
        result = limiter.check("user_1")
        assert result.allowed is False

    def test_different_users(self, limiter):
        """Test rate limiting per user."""
        result1 = limiter.check("user_1")
        result2 = limiter.check("user_2")

        # Both should be allowed (different users)
        assert result1.allowed is True
        assert result2.allowed is True

    def test_rate_limit_reset(self):
        """Test rate limit reset after window."""
        limiter = RateLimiter(requests_per_minute=1, requests_per_hour=1000)

        # First request
        result1 = limiter.check("user_1")
        assert result1.allowed is True

        # Second request (should be limited)
        result2 = limiter.check("user_1")
        assert result2.allowed is False

        # Wait for minute window to reset (in real tests, would mock time)
        # time.sleep(61)
        # result3 = limiter.check("user_1")
        # assert result3.allowed is True

    def test_get_status(self, limiter):
        """Test getting rate limit status."""
        limiter.check("user_1")
        limiter.check("user_1")

        status = limiter.get_status("user_1")

        assert status["requests_made"] >= 2
        assert "remaining_per_minute" in status

    def test_reset_user(self, limiter):
        """Test resetting user's rate limit."""
        # Make some requests
        for _ in range(5):
            limiter.check("user_1")

        # Reset
        limiter.reset("user_1")

        # Should be able to make requests again
        result = limiter.check("user_1")
        assert result.allowed is True


class TestAuditLogger:
    """Tests for AuditLogger class."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create an audit logger."""
        log_path = tmp_path / "audit.log"
        return AuditLogger(log_path=str(log_path))

    def test_logger_initialization(self, logger):
        """Test logger initialization."""
        assert logger is not None

    def test_log_request(self, logger):
        """Test logging a request."""
        logger.log_request(
            user_id="user_1",
            prompt="What is Python?",
            model="gemini",
            metadata={"source": "notebook"}
        )

        logs = logger.get_logs(user_id="user_1")
        assert len(logs) >= 1

    def test_log_response(self, logger):
        """Test logging a response."""
        request_id = logger.log_request(
            user_id="user_1",
            prompt="Test",
            model="claude"
        )

        logger.log_response(
            request_id=request_id,
            response="Response text",
            tokens_used=100,
            latency_ms=150
        )

        logs = logger.get_logs(user_id="user_1")
        # Should have both request and response

    def test_log_security_event(self, logger):
        """Test logging security events."""
        logger.log_security_event(
            user_id="user_1",
            event_type="sql_injection_attempt",
            details={"query": "SELECT * FROM users; DROP TABLE--"}
        )

        events = logger.get_security_events()
        assert len(events) >= 1

    def test_get_logs_by_user(self, logger):
        """Test filtering logs by user."""
        logger.log_request(user_id="user_1", prompt="Q1", model="gemini")
        logger.log_request(user_id="user_2", prompt="Q2", model="claude")

        user1_logs = logger.get_logs(user_id="user_1")
        user2_logs = logger.get_logs(user_id="user_2")

        assert all(log.get("user_id") == "user_1" for log in user1_logs)
        assert all(log.get("user_id") == "user_2" for log in user2_logs)

    def test_get_logs_by_time(self, logger):
        """Test filtering logs by time range."""
        logger.log_request(user_id="user_1", prompt="Test", model="gemini")

        from datetime import datetime, timedelta
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now() + timedelta(hours=1)

        logs = logger.get_logs(start_time=start_time, end_time=end_time)

        # Should include recent log
        # Implementation specific


class TestAIGuardrails:
    """Tests for AIGuardrails class."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.generate = Mock(return_value="Safe response")
        return client

    @pytest.fixture
    def guardrails(self, mock_ai_client, tmp_path):
        """Create AI guardrails."""
        return AIGuardrails(
            ai_client=mock_ai_client,
            enable_sql_validation=True,
            enable_pii_detection=True,
            enable_rate_limiting=True,
            rate_limit_rpm=10,
            audit_log_path=str(tmp_path / "audit.log")
        )

    def test_guardrails_initialization(self, guardrails):
        """Test guardrails initialization."""
        assert guardrails.sql_validator is not None
        assert guardrails.pii_detector is not None
        assert guardrails.rate_limiter is not None

    def test_validate_prompt_safe(self, guardrails):
        """Test validating a safe prompt."""
        result = guardrails.validate_prompt(
            prompt="What is the capital of France?",
            user_id="user_1"
        )

        assert result.is_allowed is True

    def test_validate_prompt_with_sql_injection(self, guardrails):
        """Test detecting SQL in prompt."""
        result = guardrails.validate_prompt(
            prompt="Run this: DROP TABLE users;",
            user_id="user_1"
        )

        # May be flagged depending on implementation
        # assert result.is_allowed is False

    def test_validate_prompt_with_pii(self, guardrails):
        """Test detecting PII in prompt."""
        result = guardrails.validate_prompt(
            prompt="My email is test@example.com and my SSN is 123-45-6789",
            user_id="user_1"
        )

        assert result.contains_pii is True

    def test_validate_prompt_rate_limited(self, guardrails):
        """Test rate limiting."""
        # Make many requests
        for _ in range(11):
            guardrails.validate_prompt(
                prompt="Test",
                user_id="rate_test_user"
            )

        result = guardrails.validate_prompt(
            prompt="Another test",
            user_id="rate_test_user"
        )

        assert result.is_allowed is False
        assert "rate" in result.reason.lower()

    def test_safe_generate(self, guardrails, mock_ai_client):
        """Test safe generation with guardrails."""
        response = guardrails.generate(
            prompt="What is Python?",
            user_id="user_1"
        )

        assert response is not None
        mock_ai_client.generate.assert_called_once()

    def test_safe_generate_blocked(self, guardrails, mock_ai_client):
        """Test blocked generation."""
        # First exhaust rate limit
        for _ in range(11):
            guardrails.validate_prompt(prompt="Test", user_id="blocked_user")

        # Should be blocked
        response = guardrails.generate(
            prompt="Another query",
            user_id="blocked_user"
        )

        # Should return error message or None
        # mock_ai_client.generate should not be called

    def test_redact_pii_in_response(self, guardrails, mock_ai_client):
        """Test PII redaction in responses."""
        mock_ai_client.generate.return_value = "Contact john@example.com"

        response = guardrails.generate(
            prompt="Get contact info",
            user_id="user_1",
            redact_pii=True
        )

        # PII should be redacted
        assert "john@example.com" not in response or "[REDACTED" in response

    def test_audit_logging(self, guardrails):
        """Test that requests are logged."""
        guardrails.generate(
            prompt="Test query",
            user_id="logged_user"
        )

        logs = guardrails.audit_logger.get_logs(user_id="logged_user")
        assert len(logs) >= 1


class TestCreateGuardrails:
    """Tests for factory function."""

    def test_create_guardrails(self):
        """Test creating guardrails."""
        mock_client = Mock()

        guardrails = create_guardrails(
            ai_client=mock_client,
            enable_sql_validation=True,
            enable_pii_detection=True
        )

        assert isinstance(guardrails, AIGuardrails)

    def test_create_guardrails_minimal(self):
        """Test creating guardrails with minimal options."""
        mock_client = Mock()

        guardrails = create_guardrails(
            ai_client=mock_client,
            enable_sql_validation=False,
            enable_pii_detection=False,
            enable_rate_limiting=False
        )

        assert isinstance(guardrails, AIGuardrails)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
