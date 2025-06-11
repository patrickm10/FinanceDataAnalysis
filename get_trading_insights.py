import argparse
import logging
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import List
import os
import polars as pl


def read_csv_to_polars(file_path: str) -> pl.DataFrame:
    """
    Reads a CSV file and returns a Polars DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the data from the CSV file.
    """
    df = pl.read_csv(
        file_path,
    )

    return df


def query_dataframe(df: pl.DataFrame, columns, values) -> pl.DataFrame:
    """
    Queries the DataFrame for rows where the specified column(s) match the given value(s),
    automatically casting string inputs to the appropriate column types.

    Args:
        df (pl.DataFrame): The Polars DataFrame to query.
        columns (str | list[str]): The column name(s) to filter on.
        values (str | list[str]): The value(s) to match in the specified column(s).

    Returns:
        pl.DataFrame: A DataFrame containing the filtered rows.
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(values, str):
        values = [values]

    if len(columns) != len(values):
        raise ValueError("Number of columns and values must match.")

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    expressions = []
    for col, val in zip(columns, values):
        dtype = df.schema[col]
        try:
            if dtype in [pl.Int32, pl.Int64]:
                cast_val = int(val)
            elif dtype == pl.Float64:
                cast_val = float(val)
            elif dtype == pl.Utf8:
                cast_val = str(val)
            else:
                raise ValueError(f"Unsupported data type: {dtype} for column '{col}'")

            # Build filter expression:
            if dtype == pl.Utf8:
                # Partial match (case sensitive)
                expressions.append(pl.col(col).str.contains(cast_val))
            else:
                # Exact match for numeric columns
                expressions.append(pl.col(col) == cast_val)

        except ValueError as e:
            raise ValueError(
                f"Value '{val}' cannot be cast to {dtype} for column '{col}'"
            ) from e

    # Combine all conditions with AND
    condition = reduce(lambda acc, expr: acc & expr, expressions)

    filtered_df = df.filter(condition)

    if filtered_df.is_empty():
        print(f"No rows found for conditions: {list(zip(columns, values))}.")
    else:
        print(
            f"Found {filtered_df.shape[0]} matching records for the following search criteria: {list(zip(columns, values))}."
        )



def find_duplicate_records(df: pl.DataFrame) -> pl.DataFrame:
    """
    Finds duplicate records in the DataFrame based on all columns.

    Args:
        df (pl.DataFrame): The Polars DataFrame to check for duplicates.

    Returns:
        pl.DataFrame: A DataFrame containing the duplicate rows.
    """
    duplicates = df.filter(df.is_duplicated())

    if duplicates.is_empty():
        print("No duplicate records found.")
    else:
        print(f"Found {duplicates.shape[0]} duplicate records.")
        print(str(f"{duplicates}\n"))
    return duplicates


def find_duplicate_records_by_columns(
    df: pl.DataFrame, columns: List[str]
) -> pl.DataFrame:
    """
    Finds duplicate records in the DataFrame based on specified columns.

    Args:
        df (pl.DataFrame): The Polars DataFrame to check for duplicates.
        columns (List[str]): List of column names to check for duplicates.

    Returns:
        pl.DataFrame: A DataFrame containing the duplicate rows based on specified columns.
    """
    if not all(col in df.columns for col in columns):
        raise ValueError("One or more specified columns do not exist in the DataFrame.")

    duplicate_groups = (
        df.group_by(columns)
        .agg(pl.len().alias("group_size"))
        .filter(pl.col("group_size") > 1)
        .select(columns)
    )

    # Joins original dataframe with duplicate groups on the specified columns to get all duplicate rows
    duplicates = df.join(duplicate_groups, on=columns, how="inner")

    if duplicates.is_empty():
        print(f"No duplicate records found based on columns: {columns}.")
    else:
        print(
            f"Found {duplicates.shape[0]} duplicate records based on columns: {columns}.\n"
        )
        print(f"Duplicate records:\n{duplicates.select(columns)}\n")

    return duplicates


def save_to_parquet(df: pl.DataFrame, output_path: str) -> None:
    """
    Saves the DataFrame to a Parquet file, creating parent directories if needed.

    Args:
        df (pl.DataFrame): The DataFrame to save.
        output_path (str): The file path where the Parquet file will be written.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.write_parquet(output_path)
    print(f"DataFrame successfully saved to Parquet at: {output_path}")


def validate_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validates the DataFrame, checking for:
    - Required fields not missing
    - Numeric columns properly parsed
    - Duplicate TradeIDs

    Returns:
        A cleaned and validated DataFrame.
    """
    required_fields = [
        "TradeDate",
        "TradeID",
        "CounterpartyID",
        "ISIN",
        "SEDOL",
        "Quantity",
        "Price",
        "Currency",
        "UpdatedAt",
    ]
    for col in required_fields:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Cast key columns in original_df to ensure schema matches df_clean for join
    original_df = df.clone().with_columns([
        pl.col("TradeID").cast(pl.Int32, strict=False),
        pl.col("CounterpartyID").cast(pl.Int32, strict=False),
        pl.col("Price").cast(pl.Float64, strict=False),
        pl.col("UpdatedAt").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M", strict=False),
    ])

    # Cast numeric columns in df for validation and cleaning
    df = df.with_columns([
        pl.col("TradeID").cast(pl.Int32, strict=False),
        pl.col("CounterpartyID").cast(pl.Int32, strict=False),
        pl.col("Quantity").cast(pl.Float64, strict=False),
        pl.col("Price").cast(pl.Float64, strict=False),
    ])

    # Fill null Quantity and Price with 0
    df_filled = df.with_columns([
        pl.col("Quantity").fill_null(0),
        pl.col("Price").fill_null(0.0),
    ])

    # Drop rows missing required non-numeric fields
    df_clean = df_filled.drop_nulls(
        subset=[col for col in required_fields if col not in ["Quantity", "Price"]]
    )

    # Parse datetime columns in df_clean
    df_clean = df_clean.with_columns([
        pl.col("TradeDate").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False),
        pl.col("UpdatedAt").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M", strict=False),
    ])

    # Drop rows where datetime parsing failed
    df_clean = df_clean.drop_nulls(subset=["TradeDate", "UpdatedAt"])

    # Filter for valid data ranges
    now = datetime.now()
    df_clean = df_clean.filter(
        (pl.col("TradeDate") <= now) &
        (pl.col("UpdatedAt") <= now) &
        (pl.col("Price") >= 0) &
        (pl.col("Quantity") >= 0) &
        (pl.col("TradeID") >= 0) &
        (pl.col("CounterpartyID") >= 0) &
        (pl.col("TradeID") <= 1000) &
        (pl.col("CounterpartyID") <= 1000)
    )

    valid_currencies = {"USD", "EUR", "GBP", "JPY", "CAD"}
    df_clean = df_clean.filter(pl.col("Currency").is_in(valid_currencies))

    id_cols = ["TradeID", "CounterpartyID", "ISIN", "Price", "UpdatedAt"]
    removed_df = original_df.join(df_clean, on=id_cols, how="anti")
    print(removed_df)

    return df_clean




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acadian Assessment")
    parser.add_argument("csv_file", type=str)
    parser.add_argument("--no-save", action="store_true", help="Skip saving to Parquet")
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Choose output format (default: parquet)",
    )

    args = parser.parse_args()

    file_path = Path(args.csv_file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}\n")

    df = read_csv_to_polars(str(file_path))
    df = validate_dataframe(df)

    query_counter_party_id = query_dataframe(df, "CounterpartyID", "5")
    print(f"{query_counter_party_id}\n")

    duplicates = find_duplicate_records(df)
    duplicate_columns = ["TradeID", "CounterpartyID", "ISIN", "Price", "UpdatedAt"]
    duplicates_by_columns = find_duplicate_records_by_columns(df, duplicate_columns)

    # Save the cleaned DataFrame based on output format
    if not args.no_save:
        if args.output_format == "parquet":
            output_path = "Parquet/trades.parquet"
            save_to_parquet(df, output_path)

            parquet_df = pl.read_parquet(output_path)
            query_multi_category = query_dataframe(
                parquet_df,
                ["CounterpartyID", "ISIN", "Price"],
                ["5", "US0378331005", "100"],
            )
            print(f"{query_multi_category}\n")

        elif args.output_format == "csv":
            output_path = "CSV/trades_cleaned.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.write_csv(output_path)
            print(f"DataFrame successfully saved to CSV at: {output_path}")

            csv_df = read_csv_to_polars(output_path)
            query_multi_category = query_dataframe(
                csv_df,
                ["CounterpartyID", "ISIN", "Price"],
                ["5", "US0378331005", "100"],
            )
            print(f"{query_multi_category}\n")
    else:
        print("Skipping save and query-from-disk due to --no-save flag.")

