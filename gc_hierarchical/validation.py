from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def _is_boolean_like(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "false", "1", "0", "yes", "no"})


def validate_dataframe(
    frame: pd.DataFrame,
    schema: dict[str, Any],
    source: Path,
) -> list[str]:
    errors: list[str] = []
    required = schema.get("required_columns", {})

    missing = [column for column in required if column not in frame.columns]
    if missing:
        errors.append("missing columns: " + ", ".join(missing))
        return errors

    for column, expected in required.items():
        series = frame[column]
        if expected in {"number", "number_or_missing"}:
            converted = pd.to_numeric(series, errors="coerce")
            if expected == "number":
                invalid = converted.isna()
            else:
                invalid = converted.isna() & series.notna()
            if invalid.any():
                errors.append(
                    f"column '{column}' contains {int(invalid.sum())} non-numeric values"
                )
        elif expected == "boolean_like":
            nonmissing = series.notna()
            invalid = nonmissing & ~_is_boolean_like(series)
            if invalid.any():
                errors.append(
                    f"column '{column}' contains {int(invalid.sum())} non-boolean values"
                )

    for column, limits in schema.get("ranges", {}).items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        lo, hi = limits
        outside = (finite < lo) | (finite > hi)
        if outside.any():
            errors.append(
                f"column '{column}' has {int(outside.sum())} values outside [{lo}, {hi}]"
            )

    allowed = schema.get("allowed_rank_values")
    if allowed and "Rank" in frame:
        values = frame["Rank"].astype(str).str.lower().str.strip()
        invalid = ~values.isin(allowed)
        if invalid.any():
            examples = ", ".join(sorted(values[invalid].unique())[:5])
            errors.append(
                f"column 'Rank' has {int(invalid.sum())} invalid values: {examples}"
            )

    return errors


def require_valid_dataframe(
    frame: pd.DataFrame,
    schema: dict[str, Any],
    source: Path,
) -> None:
    errors = validate_dataframe(frame, schema, source)
    if errors:
        detail = "\n  - ".join(errors)
        raise DataValidationError(f"Validation failed for {source}:\n  - {detail}")
