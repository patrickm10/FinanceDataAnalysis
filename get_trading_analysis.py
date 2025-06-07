import argparse
import csv
import logging
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
    with open(
        file_path, "r", encoding="utf-8-sig"
    ) as f:  #   utf-8-sig handles BOM in CSV files
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]

    return pl.DataFrame(data, schema=header, orient="row")


def query_dataframe(df: pl.DataFrame, columns, values) -> pl.DataFrame:
    """
    Queries the DataFrame for rows where the specified column(s) match the given value(s).

    Args:
        df (pl.DataFrame): The Polars DataFrame to query.
        columns (str | list[str]): The column name(s) to filter on.
        values (str | list[str]): The value(s) to match in the specified column(s).

    Returns:
        pl.DataFrame: A DataFrame containing the filtered rows.
    """
    # Creates a list for columns and values
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(values, str):
        values = [values]

    if len(columns) != len(values):
        raise ValueError("Number of columns and values must match.")

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    # Builds compound condition expression dynamically
    condition = reduce(
        lambda acc, expr: acc & expr,
        [pl.col(col) == val for col, val in zip(columns, values)],
    )

    filtered_df = df.filter(condition)

    if filtered_df.is_empty():
        print(f"No rows found for conditions: {list(zip(columns, values))}.")
    else:
        print(
            f"Found {filtered_df.shape[0]} matching records for the following search criteria: {list(zip(columns, values))}."
        )

    return filtered_df


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


def find_duplicate_records_by_columns(df: pl.DataFrame, columns: List[str]
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

    # Drops the rows with missing essential data
    df_clean = df.drop_nulls(subset=required_fields)

    print(f"Validated DataFrame with {df_clean.shape[0]} rows after cleaning.\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finance Analysis")
    parser.add_argument("csv_file", type=str)
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

    # Saves to Parquet for Scalability
    parquet_output_path = "Parquet/trades.parquet"
    save_to_parquet(df, parquet_output_path)

    parquet_df = pl.read_parquet(parquet_output_path)

    query_multi_category = query_dataframe(
        parquet_df, ["CounterpartyID", "ISIN", "Price"], ["5", "US0378331005", "100"]
    )
    print(f"{query_multi_category}\n")
