"""
great-validations: A lightweight data validation library for pandas DataFrames.
"""

import pandas as pd
from datetime import datetime
from enum import Enum
from typing import Callable, Optional
import json


class Severity(Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Rule:
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
        Returns a result dict with status, violations, and any errors encountered.
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
    def __init__(self):
        self.rules = []

    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        return self

    def validate(self, df: pd.DataFrame) -> 'ValidationReport':
        results = [rule.check(df) for rule in self.rules]
        return ValidationReport(results, len(df))


class ValidationReport:
    def __init__(self, results: list, total_rows: int):
        self.results = results
        self.total_rows = total_rows
        self.timestamp = datetime.now()

    @property
    def passed(self) -> bool:
        """Returns True only if all rules passed (no failures or errors)."""
        return all(r['status'] == 'passed' for r in self.results)

    @property
    def summary(self) -> dict:
        """High-level summary of validation run."""
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
        """Returns only failed or errored rules."""
        return [r for r in self.results if r['status'] != 'passed']

    def by_severity(self, severity: Severity) -> list:
        """Filter results by severity level."""
        return [r for r in self.results if r['severity'] == severity.value]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for analysis or export."""
        df = pd.DataFrame(self.results)
        df['columns'] = df['columns'].apply(lambda x: ', '.join(x) if x else '')
        return df

    def to_json(self, path: Optional[str] = None) -> str:
        """Export to JSON string or file."""
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
        """Generate a markdown report for sharing via Slack, email, etc."""
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

def not_null(column):
    """Check that a column has no null values."""
    return lambda df: df[column].notna()


def unique_values(column, keep='first'):
    """Check that a column has unique values."""
    return lambda df: ~df.duplicated(subset=[column], keep=keep)


def between(column, min_val, max_val):
    """Check that values in a column fall within a range (inclusive)."""
    return lambda df: df[column].between(min_val, max_val)


def in_set(column, valid_values):
    """Check that values in a column are within a set of valid values."""
    return lambda df: df[column].isin(valid_values)


def matches_pattern(column, regex):
    """Check that string values in a column match a regex pattern."""
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
