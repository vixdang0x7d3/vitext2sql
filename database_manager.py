import sqlite3
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut


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
        if sqlite_path not in self.conns:
            uri = f"file:{sqlite_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self.conns[sqlite_path] = conn

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

    def _exec_query_sqlite(
        self,
        query: str,
        sqlite_path: str,
        save_path: str,
        max_len: int,
    ) -> pd.DataFrame | None:
        """
        Main logic for query execution
        """

        conn = self.conns[sqlite_path]
        cursor = conn.cursor()

        try:
            data = pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            try:
                cursor.close()
            except Exception as e:
                print(f"Failed to close cursor: {e}")
                return None

        if data is None or data.empty:
            return None

        if save_path:
            data.to_csv(save_path)

        return data.head(max_len)

    def exec_query_sqlite(
        self,
        query: str,
        sqlite_path: str,
        max_len: int = 30_000,
        save_path: str | None = None,
        timeout: int = 300,
    ) -> pd.DataFrame | None:
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
            print(f"Query could not complete within {timeout} seconds")
            return None
        except Exception as e:
            print(f"Query `{query}` execution error: {e}")
            return None

    def get_schema(
        self,
        sqlite_path: str,
        add_sample_rows: bool = False,
        add_description: bool = False,
    ) -> tuple[list[str], list[dict]] | None:
        """
        Extract and return a list of table informations for a given sqlite database

        Parameters:
        sqlite_path: str
            Path to the target database
        add_sample_rows: bool
            Option to add sample rows to the table information
        add_description: bool
            Option to include description
        """

        fetch_ddls = "SELECT name, sql FROM sqlite_master WHERE type='table'"
        fetch_table_info = "PRAGMA table_info({})"

        conn = self.conns[sqlite_path]
        cursor = conn.cursor()
        cursor.execute(fetch_ddls)
        tables = cursor.fetchall()

        table_names = [table[0] for table in tables]

        table_schemas = []
        for table in tables:
            table_schema = {}

            # store table name
            table_name = table[0]
            table_schema["table_fullname"] = table_name

            # extract column info
            cursor.execute(fetch_table_info.format(table_name))
            column_info = cursor.fetchall()

            column_names = []
            column_types = []

            for col in column_info:
                # cid|name|type|notnull|dflt_value|pk
                column_names.append(col[1])
                column_types.append(col[2])

            table_schema["column_names"] = column_names
            table_schema["column_types"] = column_types

            if not table_schema["column_names"]:
                print(f"WARN: table {table_name} - no column")

            # extract sample values
            sample_rows = []
            if add_sample_rows:
                sampling_query = f"SELECT * FROM {table_name} LIMIT 3"
                cursor.execute(sampling_query)
                sample_rows = cursor.fetchall()

            table_schema["sample_rows"] = str(sample_rows)
            table_schemas.append(table_schema)

        return table_names, table_schemas
