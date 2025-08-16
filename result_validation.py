from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils import (
    get_values_from_table,
    deseripress_df,
)

from prompts import consistency_prompt


@dataclass
class ValidationResult:
    is_empty: bool = False
    is_valid: bool = False
    nested_values: list[str] = field(default_factory=list)
    empty_columns: list[str] = field(default_factory=list)
    processed_data: dict = field(default_factory=dict)
    normalized_df_str: str = "empty"


@dataclass
class ConsistencyState:
    """Simple state container for consistency checking"""

    results_tables: list[dict]
    results_values: list[str]

    @classmethod
    def create_empty(cls) -> "ConsistencyState":
        return cls(results_values=[], results_tables=[])


def find_nested_values(df: pd.DataFrame) -> list[str]:
    """Find values containing newlines in the dataframe"""
    nested_values = []
    for row in df.values.tolist():
        for item in row:
            if isinstance(item, str) and "\n" in item:
                nested_values.append(item)

    return nested_values


def normalize_df(df: pd.DataFrame) -> str:
    """Normalize dataframe by rounding floats and sorting by first column"""
    df_copy = df.copy()

    for col in df.select_dtypes(include=["float"]):
        df_copy[col] = df[col].round(2)

    # Sort by first column and return a string
    sort_col = df_copy.columns[0]
    df_copy_sorted = df_copy[sort_col].astype(str)
    return df_copy_sorted.to_string()


def find_empty_columns(df: pd.DataFrame) -> list[str]:
    df_str = df.astype(str)
    empty_columns = df_str.columns[
        ((df_str == "0") | (df_str == "") | (df_str == "nan")).all()
    ].tolist()
    return empty_columns


def validate_compressed_result(compressed_data: dict) -> ValidationResult:
    """
    Validate compressed DataFrame data and return validation results

    Args:
        compressed_data: Compressed dict from seripress
    """

    # Decompress to work with DataFrame
    df = deseripress_df(compressed_data)
    if df.empty:
        return ValidationResult(
            is_empty=True,
            is_valid=True,
        )

    nested_values = find_nested_values(df)
    normalize_df_str = get_values_from_table(df)
    empty_columns = find_empty_columns(df)

    is_valid = not nested_values and not empty_columns

    return ValidationResult(
        is_valid=is_valid,
        nested_values=nested_values,
        empty_columns=empty_columns,
        processed_data=compressed_data,
        normalized_df_str=normalize_df_str,
    )


def check_consistency(
    validation_result: ValidationResult,
    state: ConsistencyState,
) -> bool:
    return validation_result.normalized_df_str in state.results_values


def add_result_to_state(
    validation_result: ValidationResult,
    state: ConsistencyState,
) -> ConsistencyState:
    """Add a new result to the consistency state if it's not already present"""

    if validation_result.normalized_df_str not in state.results_values:
        new_values = state.results_values + [validation_result.normalized_df_str]
        new_tables = state.results_tables + [validation_result.processed_data]
        return ConsistencyState(
            results_values=new_values,
            results_tables=new_tables,
        )

    return state


def build_refinement_prompt(
    validation_result: ValidationResult,
    question: str,
    response: str,
    db_type: str,
) -> str:
    """Build a refinement prompt for fixing validation issues"""

    prompt = consistency_prompt.format(
        question=question,
        csv_format="",
    )

    df = deseripress_df(validation_result.processed_data)

    if not df.empty:
        prompt += f"Current answer: \n{df.to_csv(index=False)}"
    else:
        prompt += "Current answer: Empty"

    prompt += f"Current SQL:\n{response}"

    if validation_result.nested_values:
        prompt += f"Values {validation_result.nested_values} are nested. Please correct them. e.g. Transfer '[\nA,\n B\n]' to 'A, B'.\n"

    if validation_result.empty_columns:
        prompt += f"Empty results in Column {validation_result.empty_columns}. Please correct them.\n"

    return prompt


def process_result(
    compressed_data: dict[str, Any],
    question: str,
    response: str,
    db_type: str,
    consistency_state: ConsistencyState,
) -> tuple[bool, str | None, ConsistencyState]:
    """
    Process a result and return success status, refinement prompt, and update state

    Args:
        compressed_data: compressed query result
        question: natural language query
        format_csv: format of the query result to show in prompt
        response: the sql query previousy generated
        db_type: type of database engine (sqlite, pgsql, duckdb)

    Returns:
        tuple of (success, refinement_prompt, updated_state)
    """

    validation_result = validate_compressed_result(compressed_data)

    if check_consistency(validation_result, consistency_state):
        return True, None, consistency_state

    if validation_result.is_valid:
        updated_state = add_result_to_state(validation_result, consistency_state)
        return False, None, updated_state

    refinement_prompt = build_refinement_prompt(
        validation_result,
        question,
        response,
        db_type,
    )

    return False, refinement_prompt, consistency_state
