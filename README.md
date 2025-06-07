# FinanceDataAnalysis

## Dependencies:

• polars

## How to run:

• python get_trading_analysis.py "Acadian_Assessment/CSV/Sample Trade Data.csv"

## How the script works
• Scripts starts by ingesting the data as a Polars DataFrame and cleaning the null records

• Queries the dataframe for records with CounterpartyID = 5

• Filters by multiple categories next using COunterpartyID, ISIN, Price

• Searches for duplicate records based on all columns

• Searches for duplicate records based on columns TradeID, CounterpartyID, ISIN, Price, UpdatedAt

• Saves to Parquet format (Scalable).

## Data validation:

• Checks for required fields

• Drops null or malformed rows

## Duplicate detection:
• Full row duplicate search

• Customizable duplicate detection by specific fields (e.g., TradeID, ISIN, etc.)

## Flexible querying:

• Supports filtering by one or multiple columns

• Prints meaningful feedback for missing matches

## Extensibility:

• Includes optional support for saving the data to a Parquet file

• Modular and testable design using reusable functions
