"""
great-validations: A lightweight data validation library for pandas DataFrames.
"""

import pandas as pd
from datetime import datetime
from enum import Enum
from typing import Callable, Literal, Optional
import json


class Severity(Enum):
    """
    Severity levels for validation rules.

    Use these to categorize rules by importance and filter results accordingly.

    Attributes:
        WARNING: Minor issues that don't block data processing.
        ERROR: Standard validation failures that should be addressed.
        CRITICAL: Severe data integrity issues requiring immediate attention.

    Example:
        >>> rule = Rule(
        ...     name="id_not_null",
        ...     condition=not_null('id'),
        ...     error_msg="ID cannot be null",
        ...     severity=Severity.CRITICAL
        ... )
        >>> report.by_severity(Severity.CRITICAL)  # Filter critical issues
    """

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Rule:
    """
    A validation rule that checks a condition against a DataFrame.

    Rules define what constitutes valid data. Each rule has a condition (a callable
    that returns a boolean Series), an error message, and a severity level.

    Args:
        name: Unique identifier for the rule.
        condition: A callable that takes a DataFrame and returns a boolean Series
            where True indicates valid rows and False indicates violations.
        error_msg: Human-readable message describing what failed.
        severity: How serious a violation is. Defaults to Severity.ERROR.
        columns: List of column names this rule applies to (for documentation).

    Example:
        >>> # Using a helper function
        >>> rule = Rule(
        ...     name="age_in_range",
        ...     condition=between('age', 0, 120),
        ...     error_msg="Age must be between 0 and 120",
        ...     severity=Severity.ERROR,
        ...     columns=['age']
        ... )

        >>> # Using a custom lambda
        >>> rule = Rule(
        ...     name="email_valid",
        ...     condition=lambda df: df['email'].str.contains('@', na=False),
        ...     error_msg="Email must contain @",
        ...     columns=['email']
        ... )

        >>> # Check against a DataFrame
        >>> result = rule.check(df)
        >>> result['status']  # 'passed', 'failed', or 'error'
    """

    def __init__(
        self,
        name: str,
        condition: Callable,
        error_msg: str,
        severity: Severity = Severity.ERROR,
        columns: Optional[list] = None
    ):
        self.name = name
        self.condition = condition
        self.error_msg = error_msg
        self.severity = severity
        self.columns = columns or []

    def check(self, df: pd.DataFrame) -> dict:
        """
        Run this rule against a DataFrame.

        Args:
            df: The pandas DataFrame to validate.

        Returns:
            A dict containing:
                - rule: The rule name
                - severity: The severity level
                - message: The error message
                - columns: Columns this rule applies to
                - status: 'passed', 'failed', or 'error'
                - violations: Number of rows that failed
                - violation_pct: Percentage of rows that failed
                - total_rows: Total rows checked
                - sample_indices: Up to 5 indices of violating rows
                - error: Error message if status is 'error', else None

        Example:
            >>> rule = Rule("not_null", not_null('id'), "ID required")
            >>> result = rule.check(df)
            >>> if result['status'] == 'failed':
            ...     print(f"{result['violations']} rows have null IDs")
        """
        result = {
            'rule': self.name,
            'severity': self.severity.value,
            'message': self.error_msg,
            'columns': self.columns,
            'status': 'passed',
            'violations': 0,
            'violation_pct': 0.0,
            'total_rows': len(df),
            'sample_indices': [],
            'error': None
        }

        try:
            mask = self.condition(df)
            violations = df[~mask]

            if not violations.empty:
                result['status'] = 'failed'
                result['violations'] = len(violations)
                result['violation_pct'] = round(100 * len(violations) / len(df), 2)
                result['sample_indices'] = violations.head(5).index.tolist()

        except KeyError as e:
            result['status'] = 'error'
            result['error'] = f"Column not found: {e}"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = f"{type(e).__name__}: {e}"

        return result


class DataValidator:
    """
    Manages a collection of validation rules and runs them against DataFrames.

    The validator collects rules and executes them all when validate() is called,
    returning a ValidationReport with the results.

    Example:
        >>> import pandas as pd
        >>> from great_validations import DataValidator, Rule, Severity, not_null, between

        >>> df = pd.DataFrame({
        ...     'user_id': [1, 2, None, 4],
        ...     'age': [25, 150, 30, -5]
        ... })

        >>> validator = DataValidator()
        >>> validator.add_rule(Rule(
        ...     name="user_id_required",
        ...     condition=not_null('user_id'),
        ...     error_msg="User ID cannot be null",
        ...     severity=Severity.CRITICAL
        ... ))
        >>> validator.add_rule(Rule(
        ...     name="valid_age",
        ...     condition=between('age', 0, 120),
        ...     error_msg="Age must be between 0 and 120"
        ... ))

        >>> report = validator.validate(df)
        >>> report.passed  # False
        >>> print(report.to_markdown())
    """

    def __init__(self):
        self.rules = []

    def add_rule(self, rule: Rule):
        """
        Add a rule to the validator.

        Args:
            rule: The Rule to add.

        Returns:
            self, allowing method chaining.

        Example:
            >>> validator = DataValidator()
            >>> validator.add_rule(rule1).add_rule(rule2).add_rule(rule3)
        """
        self.rules.append(rule)
        return self

    def validate(self, df: pd.DataFrame) -> 'ValidationReport':
        """
        Run all rules against a DataFrame.

        Args:
            df: The pandas DataFrame to validate.

        Returns:
            A ValidationReport containing results for all rules.

        Example:
            >>> report = validator.validate(df)
            >>> if not report.passed:
            ...     print(report.to_markdown())
        """
        results = [rule.check(df) for rule in self.rules]
        return ValidationReport(results, len(df))


class ValidationReport:
    """
    Results from running validation rules against a DataFrame.

    Provides multiple ways to inspect, filter, and export validation results.

    Attributes:
        results: List of result dicts from each rule check.
        total_rows: Number of rows that were validated.
        timestamp: When the validation was run.
        passed: True if all rules passed (property).
        summary: Dict with counts and overall status (property).

    Example:
        >>> report = validator.validate(df)
        >>> report.passed  # False
        >>> report.summary
        {'timestamp': '2024-01-15T10:30:00', 'total_rows': 1000,
         'rules_checked': 5, 'passed': 3, 'failed': 2, 'errors': 0,
         'overall_status': 'FAILED'}

        >>> # Get only failures
        >>> for failure in report.failures_only():
        ...     print(f"{failure['rule']}: {failure['violations']} violations")

        >>> # Export results
        >>> report.to_json('validation_report.json')
        >>> print(report.to_markdown())
    """

    def __init__(self, results: list, total_rows: int):
        self.results = results
        self.total_rows = total_rows
        self.timestamp = datetime.now()

    @property
    def passed(self) -> bool:
        """
        Check if all rules passed.

        Returns:
            True only if all rules passed (no failures or errors).

        Example:
            >>> if report.passed:
            ...     print("All validations passed!")
            ... else:
            ...     print(report.to_markdown())
        """
        return all(r['status'] == 'passed' for r in self.results)

    @property
    def summary(self) -> dict:
        """
        Get a high-level summary of the validation run.

        Returns:
            A dict containing timestamp, total_rows, rules_checked,
            passed/failed/errors counts, and overall_status.

        Example:
            >>> report.summary['failed']  # Number of failed rules
            2
            >>> report.summary['overall_status']
            'FAILED'
        """
        statuses = [r['status'] for r in self.results]
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_rows': self.total_rows,
            'rules_checked': len(self.results),
            'passed': statuses.count('passed'),
            'failed': statuses.count('failed'),
            'errors': statuses.count('error'),
            'overall_status': 'PASSED' if self.passed else 'FAILED'
        }

    def failures_only(self) -> list:
        """
        Get only the rules that failed or errored.

        Returns:
            List of result dicts where status is 'failed' or 'error'.

        Example:
            >>> for failure in report.failures_only():
            ...     print(f"{failure['rule']}: {failure['message']}")
        """
        return [r for r in self.results if r['status'] != 'passed']

    def by_severity(self, severity: Severity) -> list:
        """
        Filter results by severity level.

        Args:
            severity: The Severity level to filter by.

        Returns:
            List of result dicts matching the specified severity.

        Example:
            >>> critical_issues = report.by_severity(Severity.CRITICAL)
            >>> if critical_issues:
            ...     raise ValueError("Critical validation failures!")
        """
        return [r for r in self.results if r['severity'] == severity.value]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns:
            DataFrame with one row per rule, useful for analysis or export.

        Example:
            >>> df = report.to_dataframe()
            >>> df[df['status'] == 'failed'][['rule', 'violations']]
        """
        df = pd.DataFrame(self.results)
        df['columns'] = df['columns'].apply(lambda x: ', '.join(x) if x else '')
        return df

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Export results to JSON.

        Args:
            path: Optional file path to write JSON to.

        Returns:
            JSON string containing summary and full results.

        Example:
            >>> json_str = report.to_json()  # Get as string
            >>> report.to_json('report.json')  # Save to file
        """
        output = {
            'summary': self.summary,
            'results': self.results
        }
        json_str = json.dumps(output, indent=2, default=str)

        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str

    def to_markdown(self) -> str:
        """
        Generate a markdown-formatted report.

        Useful for sharing results via Slack, email, GitHub comments, etc.

        Returns:
            Markdown string with summary and issues table.

        Example:
            >>> print(report.to_markdown())
            # Data Validation Report
            **Timestamp:** 2024-01-15 10:30:00
            **Rows Checked:** 1,000
            **Overall Status:** FAILED
            ...
        """
        lines = [
            f"# Data Validation Report",
            f"**Timestamp:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Rows Checked:** {self.total_rows:,}",
            f"**Overall Status:** {'PASSED' if self.passed else 'FAILED'}",
            "",
            "## Summary",
            f"- Rules Passed: {self.summary['passed']}",
            f"- Rules Failed: {self.summary['failed']}",
            f"- Rules Errored: {self.summary['errors']}",
            "",
        ]

        failures = self.failures_only()
        if failures:
            lines.append("## Issues Found")
            lines.append("")
            lines.append("| Rule | Severity | Violations | % | Message |")
            lines.append("|------|----------|------------|---|---------|")

            for r in failures:
                if r['status'] == 'error':
                    lines.append(f"| {r['rule']} | ERROR | - | - | {r['error']} |")
                else:
                    lines.append(
                        f"| {r['rule']} | {r['severity'].upper()} | "
                        f"{r['violations']:,} | {r['violation_pct']}% | {r['message']} |"
                    )
        else:
            lines.append("## All rules passed!")

        return "\n".join(lines)

    def __repr__(self):
        return f"<ValidationReport: {self.summary['overall_status']} ({self.summary['passed']}/{self.summary['rules_checked']} rules passed)>"


# --- Helper functions for common validation patterns ---

def not_null(column: str) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Create a condition that checks for non-null values.

    Args:
        column: The column name to check.

    Returns:
        A condition function for use with Rule.

    Example:
        >>> rule = Rule(
        ...     name="id_required",
        ...     condition=not_null('user_id'),
        ...     error_msg="User ID cannot be null",
        ...     columns=['user_id']
        ... )
    """
    return lambda df: df[column].notna()


def unique_values(column: str, keep: Literal['first', 'last', False] = 'first') -> Callable[[pd.DataFrame], pd.Series]:
    """
    Create a condition that checks for unique values (no duplicates).

    Args:
        column: The column name to check.
        keep: Which duplicates to mark as violations.
            - 'first': Mark duplicates except the first occurrence.
            - 'last': Mark duplicates except the last occurrence.
            - False: Mark all duplicates as violations.

    Returns:
        A condition function for use with Rule.

    Example:
        >>> rule = Rule(
        ...     name="unique_email",
        ...     condition=unique_values('email'),
        ...     error_msg="Duplicate email addresses found",
        ...     columns=['email']
        ... )
    """
    return lambda df: ~df.duplicated(subset=[column], keep=keep)


def between(column: str, min_val, max_val) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Create a condition that checks if values fall within a range (inclusive).

    Args:
        column: The column name to check.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        A condition function for use with Rule.

    Example:
        >>> rule = Rule(
        ...     name="valid_age",
        ...     condition=between('age', 0, 120),
        ...     error_msg="Age must be between 0 and 120",
        ...     columns=['age']
        ... )
    """
    return lambda df: df[column].between(min_val, max_val)


def in_set(column: str, valid_values: list) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Create a condition that checks if values are in a set of allowed values.

    Args:
        column: The column name to check.
        valid_values: List of allowed values.

    Returns:
        A condition function for use with Rule.

    Example:
        >>> rule = Rule(
        ...     name="valid_status",
        ...     condition=in_set('status', ['active', 'inactive', 'pending']),
        ...     error_msg="Invalid status value",
        ...     columns=['status']
        ... )
    """
    return lambda df: df[column].isin(valid_values)


def matches_pattern(column: str, regex: str) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Create a condition that checks if string values match a regex pattern.

    Args:
        column: The column name to check.
        regex: Regular expression pattern to match against.

    Returns:
        A condition function for use with Rule.

    Example:
        >>> # Check for valid US phone format
        >>> rule = Rule(
        ...     name="valid_phone",
        ...     condition=matches_pattern('phone', r'^\\d{3}-\\d{3}-\\d{4}$'),
        ...     error_msg="Phone must be in format XXX-XXX-XXXX",
        ...     columns=['phone']
        ... )

        >>> # Check for valid email format (simple)
        >>> rule = Rule(
        ...     name="valid_email",
        ...     condition=matches_pattern('email', r'^[^@]+@[^@]+\\.[^@]+$'),
        ...     error_msg="Invalid email format",
        ...     columns=['email']
        ... )
    """
    return lambda df: df[column].astype(str).str.match(regex, na=False)


__all__ = [
    "Severity",
    "Rule",
    "DataValidator",
    "ValidationReport",
    "not_null",
    "unique_values",
    "between",
    "in_set",
    "matches_pattern",
]
