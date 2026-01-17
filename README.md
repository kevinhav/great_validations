# great-validations

A lightweight, expressive data validation library for pandas DataFrames.

## Installation

```bash
pip install great-validations
```

## Quick Start

```python
import pandas as pd
from great_validations import DataValidator, Rule, Severity, not_null, between, in_set

# Sample data
df = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'age': [25, 30, -5, 45, 200],
    'status': ['active', 'inactive', 'active', 'pending', 'unknown']
})

# Create validator and add rules
validator = DataValidator()
validator.add_rule(Rule(
    name="user_id_not_null",
    condition=not_null('user_id'),
    error_msg="User ID cannot be null",
    severity=Severity.CRITICAL,
    columns=['user_id']
))
validator.add_rule(Rule(
    name="valid_age",
    condition=between('age', 0, 120),
    error_msg="Age must be between 0 and 120",
    severity=Severity.ERROR,
    columns=['age']
))
validator.add_rule(Rule(
    name="valid_status",
    condition=in_set('status', ['active', 'inactive', 'pending']),
    error_msg="Status must be active, inactive, or pending",
    severity=Severity.WARNING,
    columns=['status']
))

# Run validation
report = validator.validate(df)

# Check results
print(report)  # <ValidationReport: FAILED (1/3 rules passed)>
print(report.passed)  # False
print(report.summary)
```

## Features

### Severity Levels

Rules can have different severity levels to help prioritize issues:

```python
from great_validations import Severity

Severity.WARNING   # Minor issues
Severity.ERROR     # Standard validation failures
Severity.CRITICAL  # Data integrity issues
```

### Built-in Condition Helpers

```python
from great_validations import not_null, unique_values, between, in_set, matches_pattern

# Check for null values
not_null('column_name')

# Check for duplicate values
unique_values('column_name')

# Check numeric range (inclusive)
between('column_name', min_val=0, max_val=100)

# Check against allowed values
in_set('column_name', ['value1', 'value2', 'value3'])

# Check regex pattern match
matches_pattern('column_name', r'^[A-Z]{2}\d{4}$')
```

### Custom Conditions

Write any condition as a lambda or function that takes a DataFrame and returns a boolean Series:

```python
# Custom condition: email must contain @
validator.add_rule(Rule(
    name="valid_email",
    condition=lambda df: df['email'].str.contains('@', na=False),
    error_msg="Invalid email format",
    columns=['email']
))

# Complex multi-column condition
validator.add_rule(Rule(
    name="end_after_start",
    condition=lambda df: df['end_date'] > df['start_date'],
    error_msg="End date must be after start date",
    columns=['start_date', 'end_date']
))
```

### ValidationReport

The validation report provides multiple ways to inspect results:

```python
report = validator.validate(df)

# Properties
report.passed          # True if all rules passed
report.summary         # Dict with counts and overall status
report.results         # Full list of rule results
report.total_rows      # Number of rows validated
report.timestamp       # When validation ran

# Methods
report.failures_only()              # Get only failed/errored rules
report.by_severity(Severity.ERROR)  # Filter by severity

# Export formats
report.to_dataframe()   # pandas DataFrame of results
report.to_json()        # JSON string
report.to_json('report.json')  # Save to file
report.to_markdown()    # Markdown formatted report
```

### Result Structure

Each rule check returns a result dictionary:

```python
{
    'rule': 'valid_age',
    'severity': 'error',
    'message': 'Age must be between 0 and 120',
    'columns': ['age'],
    'status': 'failed',      # 'passed', 'failed', or 'error'
    'violations': 2,
    'violation_pct': 40.0,
    'total_rows': 5,
    'sample_indices': [2, 4],  # First 5 violating row indices
    'error': None              # Error message if status is 'error'
}
```

## License

MIT
