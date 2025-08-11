import sqlite3
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
import os
from dataclasses import dataclass
from enum import Enum


class SQLError(Enum):
    """Simple enum for SQL error types"""

    # self-refine handled errors
    SYNTAX_ERROR = "syntax_error"
    TABLE_NOT_FOUND = "table_not_found"
    COLUMN_NOT_FOUND = "column_not_found"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    TYPE_MISMATCH = "type_mismatch"
    FOREIGN_KEY_CONSTRAINT = "foreign_key_constraint"
    UNIQUE_CONSTRAINT = "unique_constraint"
    TIMEOUT_ERROR = "timeout_error"

    # unrecoverable errors
    DATABASE_LOCKED = "database_locked"
    PERMISSION_DENIED = "permission_denied"
    CONNECTION_ERROR = "connection_error"
    EXECUTION_ERROR = "execution_error"
    SAVE_WARNING = "save_warning"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class QueryResult:
    """
    Structured result class for database queries that includes error
    information for agent self-refinement
    """

    query: str
    success: bool
    data: pd.DataFrame | None = None
    rows_affected: int | None = None
    error_type: str | None = None
    error_message: str | None = None


class DatabaseManager:
    def __init__(self, db_type: str = "sqlite", credential: dict | None = None):
        self.db_type = db_type
        self.credential = credential
        self.conns = {}

    def start_sqlite(self, sqlite_path: str):
        """
        Create a sqlite connection and add to the connection list

        Parameters:
        sqlite_path: str
            Path to the target database
        """
        normalized_path = os.path.normpath(sqlite_path)
        print(normalized_path)
        if sqlite_path not in self.conns:
            uri = f"file:{normalized_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self.conns[normalized_path] = conn

    def close_all(self):
        """
        Close all connections in the internal connection list
        """

        for key, conn in list(self.conns.items()):
            try:
                if conn:
                    conn.close()
                    del self.conns[key]
            except Exception as e:
                print(f"Failed when closing for db {key}: {e}")

    def _categorize_sql_error(self, error: Exception) -> str:
        """
        Categorize SQL errors for building better prompt at caller level
        """

        error_str = str(error).lower()
        if "syntax error" in error_str or "near" in error_str:
            return SQLError.SYNTAX_ERROR
        elif "no such table" in error_str:
            return SQLError.TABLE_NOT_FOUND
        elif "no such column" in error_str:
            return SQLError.COLUMN_NOT_FOUND
        elif "ambiguous column" in error_str:
            return SQLError.AMBIGUOUS_COLUMN
        elif "datatype mismatch" in error_str or "type mismatch" in error_str:
            return SQLError.TYPE_MISMATCH
        elif "foreign key constraint" in error_str:
            return SQLError.FOREIGN_KEY_CONSTRAINT
        elif "unique constraint" in error_str:
            return SQLError.UNIQUE_CONSTRAINT
        elif "database is locked" in error_str:
            return SQLError.DATABASE_LOCKED
        elif "permission denied" in error_str:
            return SQLError.PERMISSION_DENIED
        else:
            return SQLError.UNKNOWN_ERROR

    def _exec_query_sqlite(
        self,
        query: str,
        sqlite_path: str,
        save_path: str,
        max_len: int,
    ) -> QueryResult:
        """
        Main logic for query execution
        """
        sqlite_path = os.path.normpath(sqlite_path)
        # print(sqlite_path)
        # print(self.conns)
        if sqlite_path not in self.conns:
            return QueryResult(
                query=query,
                success=False,
                error_type=SQLError.CONNECTION_ERROR,
                error_message=f"No connection found for database: {sqlite_path}",
            )

        conn = self.conns[sqlite_path]
        cursor = conn.cursor()

        try:
            data = pd.read_sql(query, conn)

            if data is None or data.empty:
                return QueryResult(
                    success=True,
                    data=pd.DataFrame(),
                    query=query,
                    rows_affected=0,
                )

            if save_path:
                try:
                    data.to_csv(save_path, index=False)
                except Exception as save_error:
                    print(
                        f"Query succeeded but failed to save to {save_path}: {str(save_error)}"
                    )

            return QueryResult(
                success=True,
                data=data.head(max_len),
                query=query,
                rows_affected=len(data),
            )

        except Exception as e:
            error_type = self._categorize_sql_error(e)
            error_message = str(e)

            return QueryResult(
                success=False,
                error_message=error_message,
                error_type=error_type,
                query=query,
            )

        finally:
            try:
                cursor.close()
            except Exception as cursor_error:
                print(f"Failed to close cursor: {str(cursor_error)}")
                pass

    def exec_query_sqlite(
        self,
        query: str,
        sqlite_path: str,
        max_len: int = 30_000,
        save_path: str | None = None,
        timeout: int = 300,
    ) -> QueryResult:
        """
        Exposed method for executing sql query on connected database with timeout support

        Parameters:
        query: str
            SQL query to execute
        sqlite_path: str
            Path to the target database
        max_len: int
            Maximum number of rows should be returned
        save_path: str
            Optional save_path to save the query result as csv file

        Return: Dataframe
            a pandas dataframe containing query result
        """

        try:
            result = func_timeout(
                timeout,
                self._exec_query_sqlite,
                args=(query, sqlite_path, save_path, max_len),
            )
            return result

        except FunctionTimedOut:
            return QueryResult(
                query=query,
                success=False,
                error_type=SQLError.TIMEOUT_ERROR,
                error_message=(
                    f"Query execution timed out after {timeout} seconds. "
                    f"Consider optimizing the query."
                ),
            )
        except Exception as e:
            return QueryResult(
                query=query,
                success=False,
                error_type=SQLError.UNKNOWN_ERROR,
                error_message=f"Unexpected error during query execution: {str(e)}",
            )

    def format_query_result(
        self, result: QueryResult, max_rows: int = 20, max_col_width: int = 50
    ) -> str:
        """
        Format query result into a human-readable string

        Parameters:
        result: QueryResult
            The result object from query execution
        max_rows: int
            Maximum number of rows to display
        max_col_width: int
            Maximum width for each column

        Returns:
        str
            Formatted string representation of the query result
        """

        if not result.success:
            return f"""
Query Failed:
{"-" * 50}
Query: {result.query}
Error Type: {result.error_type}
Error Message: {result.error_message}
"""

        if result.data is None or result.data.empty:
            return f"""
Query Result:
{"-" * 50}
Query: {result.query}
Status: Success (No data returned)
Rows Affected: {result.rows_affected or 0}
"""

        data = result.data
        total_rows = len(data)

        # Create header
        output = f"""
Query Result:
{"-" * 50}
Query: {result.query}
Total Rows: {total_rows:,}
Columns: {len(data.columns)}
Showing: {min(max_rows, total_rows)} rows
{"-" * 50}
"""

        # Add formatted data
        display_data = data.head(max_rows)
        formatted_table = display_data.to_string(
            max_cols=None, max_colwidth=max_col_width, index=True, na_rep="NULL"
        )

        output += formatted_table

        # Add truncation notice if needed
        if total_rows > max_rows:
            output += f"\n\n... ({total_rows - max_rows:,} more rows truncated)"

        return output

    def validate_query_syntax(self, query: str, sqlite_path: str) -> QueryResult:
        """
        Validate SQL query syntax without executing it
        Useful for
        """
        sqlite_path = os.path.normpath(sqlite_path)
        if sqlite_path not in self.conns:
            return QueryResult(
                query=query,
                success=False,
                error_type=SQLError.CONNECTION_ERROR,
                error_message=f"No connection found for database: {sqlite_path}",
            )

        conn = self.conns[sqlite_path]
        cursor = conn.cursor()

        try:
            # Use EXPLAIN to validate syntax without executing
            cursor.execute(f"EXPLAIN {query}")
            return QueryResult(
                success=True,
                query=query,
            )
        except Exception as e:
            return QueryResult(
                success=False,
                error_type=self._categorize_sql_error(e),
                error_message=str(e),
                query=query,
            )
        finally:
            try:
                cursor.close()
            except Exception as cursor_error:
                print(f"Failed to close cursor: {str(cursor_error)}")
                pass
