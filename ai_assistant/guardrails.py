"""
Guardrails and Security Module for AI Assistant.

This module provides security guardrails for AI operations, including
SQL validation, PII detection, rate limiting, and audit logging.

Features:
- SQL injection prevention
- Dangerous operation blocking
- PII detection in prompts/responses
- Rate limiting per user/workspace
- Comprehensive audit logging
- Content filtering
"""

import re
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Set
from enum import Enum
from functools import wraps
from collections import defaultdict
import threading


class RiskLevel(Enum):
    """Risk level for detected issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ActionType(Enum):
    """Type of action to take on violation."""
    BLOCK = "block"
    WARN = "warn"
    LOG = "log"
    REDACT = "redact"


@dataclass
class SecurityViolation:
    """
    Represents a security violation.

    Attributes:
        rule_name: Name of the violated rule
        risk_level: Risk level of the violation
        description: Description of the violation
        matched_content: Content that triggered the violation
        action_taken: Action taken in response
        metadata: Additional metadata
    """
    rule_name: str
    risk_level: RiskLevel
    description: str
    matched_content: str = ""
    action_taken: ActionType = ActionType.LOG
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "risk_level": self.risk_level.value,
            "description": self.description,
            "matched_content": self.matched_content[:100] if self.matched_content else "",
            "action_taken": self.action_taken.value,
            "metadata": self.metadata
        }


@dataclass
class GuardrailResult:
    """
    Result of guardrail check.

    Attributes:
        passed: Whether all checks passed
        violations: List of violations found
        processed_content: Content after processing (may be redacted)
        metadata: Additional metadata
    """
    passed: bool
    violations: List[SecurityViolation] = field(default_factory=list)
    processed_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "metadata": self.metadata
        }


class SQLValidator:
    """
    Validates SQL queries for dangerous operations.

    Blocks or warns about potentially harmful SQL operations.

    Example:
        >>> validator = SQLValidator()
        >>> result = validator.validate("DROP TABLE users")
        >>> print(result.passed)  # False
    """

    # Dangerous SQL patterns
    DANGEROUS_PATTERNS = [
        (r'\bDROP\s+(TABLE|DATABASE|SCHEMA|VIEW)\b', RiskLevel.CRITICAL, "DROP operation"),
        (r'\bTRUNCATE\s+TABLE\b', RiskLevel.CRITICAL, "TRUNCATE operation"),
        (r'\bDELETE\s+FROM\b(?!\s+WHERE)', RiskLevel.HIGH, "DELETE without WHERE"),
        (r'\bUPDATE\b(?!.*\bWHERE\b)', RiskLevel.HIGH, "UPDATE without WHERE"),
        (r'\bGRANT\s+ALL\b', RiskLevel.HIGH, "GRANT ALL PRIVILEGES"),
        (r'\bALTER\s+USER\b', RiskLevel.HIGH, "ALTER USER"),
        (r'\bCREATE\s+USER\b', RiskLevel.MEDIUM, "CREATE USER"),
        (r';\s*--', RiskLevel.HIGH, "Comment after semicolon (possible injection)"),
        (r"'\s*OR\s+'1'\s*=\s*'1", RiskLevel.CRITICAL, "SQL injection pattern"),
        (r"'\s*OR\s+1\s*=\s*1", RiskLevel.CRITICAL, "SQL injection pattern"),
        (r'\bUNION\s+SELECT\b', RiskLevel.MEDIUM, "UNION SELECT (possible injection)"),
        (r'\bINTO\s+OUTFILE\b', RiskLevel.CRITICAL, "INTO OUTFILE"),
        (r'\bLOAD_FILE\b', RiskLevel.CRITICAL, "LOAD_FILE function"),
        (r'\bEXEC\s*\(', RiskLevel.CRITICAL, "EXEC function"),
        (r'\bxp_cmdshell\b', RiskLevel.CRITICAL, "xp_cmdshell"),
    ]

    # Allowed operations (whitelist)
    SAFE_OPERATIONS = {"SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "WITH"}

    def __init__(
        self,
        block_dangerous: bool = True,
        allowed_operations: Optional[Set[str]] = None,
        custom_patterns: Optional[List[tuple]] = None
    ):
        self.block_dangerous = block_dangerous
        self.allowed_operations = allowed_operations or self.SAFE_OPERATIONS
        self.patterns = self.DANGEROUS_PATTERNS.copy()

        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def validate(self, sql: str) -> GuardrailResult:
        """
        Validate SQL query.

        Args:
            sql: SQL query to validate

        Returns:
            GuardrailResult with validation outcome
        """
        violations = []
        sql_upper = sql.upper().strip()

        # Check first operation
        first_word = sql_upper.split()[0] if sql_upper else ""
        if first_word and first_word not in self.allowed_operations:
            # Check if it's a dangerous operation
            if first_word in {"DROP", "TRUNCATE", "DELETE", "UPDATE", "ALTER", "GRANT", "REVOKE"}:
                violations.append(SecurityViolation(
                    rule_name="disallowed_operation",
                    risk_level=RiskLevel.HIGH,
                    description=f"Operation '{first_word}' is not in allowed operations",
                    matched_content=first_word,
                    action_taken=ActionType.BLOCK if self.block_dangerous else ActionType.WARN
                ))

        # Check for dangerous patterns
        for pattern, risk_level, description in self.patterns:
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                violations.append(SecurityViolation(
                    rule_name="dangerous_pattern",
                    risk_level=risk_level,
                    description=description,
                    matched_content=match.group(),
                    action_taken=ActionType.BLOCK if self.block_dangerous else ActionType.WARN
                ))

        # Determine if passed
        passed = True
        if self.block_dangerous:
            passed = not any(
                v.risk_level in {RiskLevel.CRITICAL, RiskLevel.HIGH}
                for v in violations
            )

        return GuardrailResult(
            passed=passed,
            violations=violations,
            processed_content=sql,
            metadata={"query_length": len(sql)}
        )


class PIIDetector:
    """
    Detects and optionally redacts PII in text.

    Identifies common PII patterns and can redact them from
    prompts and responses.

    Example:
        >>> detector = PIIDetector()
        >>> result = detector.scan("Email: john@example.com")
        >>> print(result.violations)  # Found email
    """

    # PII patterns
    PII_PATTERNS = [
        # Email
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
         "email", RiskLevel.MEDIUM),
        # Phone (US format)
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
         "phone_number", RiskLevel.MEDIUM),
        # SSN
        (r'\b\d{3}-\d{2}-\d{4}\b',
         "ssn", RiskLevel.CRITICAL),
        # Credit Card (basic)
        (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
         "credit_card", RiskLevel.CRITICAL),
        # IP Address
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
         "ip_address", RiskLevel.LOW),
        # API Key patterns (common formats)
        (r'\b[A-Za-z0-9]{32,}\b',
         "potential_api_key", RiskLevel.HIGH),
        # AWS Access Key
        (r'\bAKIA[A-Z0-9]{16}\b',
         "aws_access_key", RiskLevel.CRITICAL),
        # Private Key
        (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
         "private_key", RiskLevel.CRITICAL),
    ]

    # Redaction placeholders
    REDACTION_MAP = {
        "email": "[EMAIL_REDACTED]",
        "phone_number": "[PHONE_REDACTED]",
        "ssn": "[SSN_REDACTED]",
        "credit_card": "[CC_REDACTED]",
        "ip_address": "[IP_REDACTED]",
        "potential_api_key": "[KEY_REDACTED]",
        "aws_access_key": "[AWS_KEY_REDACTED]",
        "private_key": "[PRIVATE_KEY_REDACTED]",
    }

    def __init__(
        self,
        redact: bool = False,
        block_on_critical: bool = True,
        custom_patterns: Optional[List[tuple]] = None
    ):
        self.redact = redact
        self.block_on_critical = block_on_critical
        self.patterns = self.PII_PATTERNS.copy()

        if custom_patterns:
            self.patterns.extend(custom_patterns)

    def scan(self, text: str) -> GuardrailResult:
        """
        Scan text for PII.

        Args:
            text: Text to scan

        Returns:
            GuardrailResult with findings
        """
        violations = []
        processed_text = text

        for pattern, pii_type, risk_level in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                violations.append(SecurityViolation(
                    rule_name=f"pii_{pii_type}",
                    risk_level=risk_level,
                    description=f"Detected {pii_type}",
                    matched_content=match.group()[:20] + "...",
                    action_taken=ActionType.REDACT if self.redact else ActionType.WARN
                ))

                # Redact if enabled
                if self.redact:
                    replacement = self.REDACTION_MAP.get(pii_type, "[REDACTED]")
                    processed_text = processed_text.replace(match.group(), replacement)

        # Determine if passed
        passed = True
        if self.block_on_critical:
            passed = not any(v.risk_level == RiskLevel.CRITICAL for v in violations)

        return GuardrailResult(
            passed=passed,
            violations=violations,
            processed_content=processed_text,
            metadata={"pii_count": len(violations)}
        )


class RateLimiter:
    """
    Rate limiting for AI operations.

    Prevents abuse by limiting requests per user/workspace.

    Example:
        >>> limiter = RateLimiter(requests_per_minute=10)
        >>> if limiter.check("user123"):
        ...     # Allow request
        >>> else:
        ...     # Rate limited
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_limit: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_limit = burst_limit

        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, user_id: str) -> bool:
        """
        Check if request is allowed.

        Args:
            user_id: User or workspace identifier

        Returns:
            True if allowed, False if rate limited
        """
        with self._lock:
            now = time.time()

            # Clean old requests
            self._requests[user_id] = [
                t for t in self._requests[user_id]
                if now - t < 86400  # Keep last 24 hours
            ]

            requests = self._requests[user_id]

            # Check limits
            minute_requests = sum(1 for t in requests if now - t < 60)
            hour_requests = sum(1 for t in requests if now - t < 3600)
            day_requests = len(requests)

            if minute_requests >= self.requests_per_minute:
                return False
            if hour_requests >= self.requests_per_hour:
                return False
            if day_requests >= self.requests_per_day:
                return False

            # Check burst (requests in last second)
            burst_requests = sum(1 for t in requests if now - t < 1)
            if burst_requests >= self.burst_limit:
                return False

            # Record request
            self._requests[user_id].append(now)
            return True

    def get_usage(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user."""
        with self._lock:
            now = time.time()
            requests = self._requests.get(user_id, [])

            return {
                "minute": sum(1 for t in requests if now - t < 60),
                "hour": sum(1 for t in requests if now - t < 3600),
                "day": len([t for t in requests if now - t < 86400]),
                "limits": {
                    "minute": self.requests_per_minute,
                    "hour": self.requests_per_hour,
                    "day": self.requests_per_day
                }
            }


class AuditLogger:
    """
    Audit logging for AI operations.

    Logs all AI operations for compliance and debugging.

    Example:
        >>> logger = AuditLogger()
        >>> logger.log_request("user123", "prompt", "response", {"model": "claude"})
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        log_to_table: bool = False,
        spark: Any = None,
        audit_table: str = "audit.ai_assistant.logs"
    ):
        self.log_file = log_file
        self.log_to_table = log_to_table
        self.spark = spark
        self.audit_table = audit_table

        # Setup Python logger
        self.logger = logging.getLogger("ai_assistant.audit")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)

    def log_request(
        self,
        user_id: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an AI request.

        Args:
            user_id: User identifier
            prompt: Request prompt (may be truncated)
            response: AI response (may be truncated)
            metadata: Additional metadata
        """
        timestamp = time.time()
        log_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": metadata or {}
        }

        # Log to Python logger
        self.logger.info(f"AI Request: {log_entry}")

        # Log to Delta table if configured
        if self.log_to_table and self.spark:
            try:
                import json
                self.spark.sql(f"""
                    INSERT INTO {self.audit_table}
                    VALUES (
                        {timestamp},
                        '{user_id}',
                        '{log_entry["prompt_hash"]}',
                        {log_entry["prompt_length"]},
                        {log_entry["response_length"]},
                        '{json.dumps(metadata or {})}'
                    )
                """)
            except Exception as e:
                self.logger.error(f"Failed to log to table: {e}")

    def log_violation(
        self,
        user_id: str,
        violation: SecurityViolation
    ) -> None:
        """Log a security violation."""
        self.logger.warning(
            f"Security Violation: user={user_id}, "
            f"rule={violation.rule_name}, "
            f"risk={violation.risk_level.value}, "
            f"action={violation.action_taken.value}"
        )


class AIGuardrails:
    """
    Comprehensive guardrails for AI operations.

    Combines SQL validation, PII detection, rate limiting,
    and audit logging.

    Args:
        sql_validator: SQL validation instance
        pii_detector: PII detection instance
        rate_limiter: Rate limiting instance
        audit_logger: Audit logging instance

    Example:
        >>> guardrails = AIGuardrails()
        >>> result = guardrails.check_prompt("user123", "SELECT * FROM users")
        >>> if result.passed:
        ...     # Proceed with AI call
        >>> else:
        ...     # Handle violations
    """

    def __init__(
        self,
        sql_validator: Optional[SQLValidator] = None,
        pii_detector: Optional[PIIDetector] = None,
        rate_limiter: Optional[RateLimiter] = None,
        audit_logger: Optional[AuditLogger] = None
    ):
        self.sql_validator = sql_validator or SQLValidator()
        self.pii_detector = pii_detector or PIIDetector()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.audit_logger = audit_logger or AuditLogger()

    def check_prompt(
        self,
        user_id: str,
        prompt: str,
        check_rate_limit: bool = True,
        check_pii: bool = True,
        check_sql: bool = False
    ) -> GuardrailResult:
        """
        Check a prompt against all guardrails.

        Args:
            user_id: User identifier
            prompt: Prompt to check
            check_rate_limit: Whether to check rate limits
            check_pii: Whether to check for PII
            check_sql: Whether to validate as SQL

        Returns:
            GuardrailResult with combined results
        """
        all_violations = []
        processed_content = prompt
        passed = True

        # Rate limiting
        if check_rate_limit:
            if not self.rate_limiter.check(user_id):
                all_violations.append(SecurityViolation(
                    rule_name="rate_limit",
                    risk_level=RiskLevel.MEDIUM,
                    description="Rate limit exceeded",
                    action_taken=ActionType.BLOCK
                ))
                passed = False

        # PII detection
        if check_pii and passed:
            pii_result = self.pii_detector.scan(prompt)
            all_violations.extend(pii_result.violations)
            processed_content = pii_result.processed_content or processed_content
            if not pii_result.passed:
                passed = False

        # SQL validation
        if check_sql and passed:
            sql_result = self.sql_validator.validate(prompt)
            all_violations.extend(sql_result.violations)
            if not sql_result.passed:
                passed = False

        # Log violations
        for violation in all_violations:
            self.audit_logger.log_violation(user_id, violation)

        return GuardrailResult(
            passed=passed,
            violations=all_violations,
            processed_content=processed_content,
            metadata={
                "user_id": user_id,
                "checks_performed": {
                    "rate_limit": check_rate_limit,
                    "pii": check_pii,
                    "sql": check_sql
                }
            }
        )

    def check_response(
        self,
        user_id: str,
        response: str,
        check_pii: bool = True
    ) -> GuardrailResult:
        """
        Check an AI response for issues.

        Args:
            user_id: User identifier
            response: AI response to check
            check_pii: Whether to check for PII

        Returns:
            GuardrailResult
        """
        all_violations = []
        processed_content = response

        if check_pii:
            pii_result = self.pii_detector.scan(response)
            all_violations.extend(pii_result.violations)
            processed_content = pii_result.processed_content or processed_content

        return GuardrailResult(
            passed=len(all_violations) == 0,
            violations=all_violations,
            processed_content=processed_content
        )


def create_guardrails(
    block_dangerous_sql: bool = True,
    redact_pii: bool = False,
    requests_per_minute: int = 60,
    audit_log_file: Optional[str] = None
) -> AIGuardrails:
    """
    Factory function to create AIGuardrails.

    Args:
        block_dangerous_sql: Whether to block dangerous SQL
        redact_pii: Whether to redact PII
        requests_per_minute: Rate limit per minute
        audit_log_file: Path to audit log file

    Returns:
        Configured AIGuardrails

    Example:
        >>> guardrails = create_guardrails(
        ...     block_dangerous_sql=True,
        ...     redact_pii=True,
        ...     requests_per_minute=30
        ... )
    """
    return AIGuardrails(
        sql_validator=SQLValidator(block_dangerous=block_dangerous_sql),
        pii_detector=PIIDetector(redact=redact_pii),
        rate_limiter=RateLimiter(requests_per_minute=requests_per_minute),
        audit_logger=AuditLogger(log_file=audit_log_file)
    )


def guardrail_protected(guardrails: AIGuardrails, user_id_arg: str = "user_id"):
    """
    Decorator to protect a function with guardrails.

    Args:
        guardrails: AIGuardrails instance
        user_id_arg: Argument name for user ID

    Example:
        >>> @guardrail_protected(guardrails, "user_id")
        ... def ask_ai(user_id: str, prompt: str):
        ...     return ai_client.generate(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get user_id and prompt
            user_id = kwargs.get(user_id_arg, "anonymous")
            prompt = kwargs.get("prompt", args[1] if len(args) > 1 else "")

            # Check guardrails
            result = guardrails.check_prompt(user_id, prompt)

            if not result.passed:
                raise PermissionError(
                    f"Request blocked: {[v.description for v in result.violations]}"
                )

            # Update prompt with processed content
            if result.processed_content != prompt:
                kwargs["prompt"] = result.processed_content

            return func(*args, **kwargs)

        return wrapper
    return decorator
