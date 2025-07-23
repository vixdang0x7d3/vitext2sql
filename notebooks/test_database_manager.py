import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os
    import time

    import pandas as pd

    # add parent directory to kernel's path
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


    from database_manager import DatabaseManager
    return DatabaseManager, mo, pd, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Test database connection management""")
    return


@app.cell
def _(DatabaseManager):
    db_path = "data/tpch-small.db"

    dbman = DatabaseManager()
    dbman.start_sqlite(db_path)
    assert db_path in dbman.conns
    print("[x] Connection established")

    initial_conn = dbman.conns[db_path]
    dbman.start_sqlite(db_path)

    assert dbman.conns[db_path] is initial_conn
    print("[x] Connection reuse works")
    return db_path, dbman


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Test schema extraction""")
    return


@app.cell
def _(db_path, dbman):
    table_names, schemas = dbman.get_schema(db_path)

    assert isinstance(table_names, list)
    assert isinstance(schemas, list)
    assert len(table_names) == len(schemas)

    print(f"[x] Found {len(table_names)} tables: {table_names}")

    # Validate schema structure
    for schema in schemas:
        assert "table_fullname" in schema
        assert "column_names" in schema
        assert "column_types" in schema
        assert "sample_rows" in schema
        assert len(schema["column_names"]) == len(schema["column_types"])

    print("[x] Schema structure valid")

    for schema in schemas:
        print(schema)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Test row sampling""")
    return


@app.cell
def _(db_path, dbman):
    def _():
        table_names_sample, schema_sample = dbman.get_schema(
            db_path, add_sample_rows=True
        )

        for i, schema in enumerate(schema_sample):
            if schema["sample_rows"] != "[]":
                print(f"[x] Table {schema['table_fullname']} has sample data")
                print(f"    Sample preview: {schema['sample_rows'][:100]}")


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Basic query execution""")
    return


@app.cell
def _(db_path, dbman):
    test_queries = [
        "select count(*) as count from customer limit 1",
        "select * from customer limit 5",
        "select c_name, c_nationkey from customer limit 3",
    ]

    for query in test_queries:
        try:
            result = dbman._exec_query_sqlite_timeout(query, db_path, max_len=10)
            if result is not None:
                print(f"[x] Query executed: {query[:50]}...")
                print(f"    Result shape: {result.shape} ")
            else:
                print(f"[!] Query failed: {query}")
        except Exception as e:
            print(f"Query error: {query} - {e}")
    return


@app.cell
def _(db_path, dbman):
    dbman._exec_query_sqlite_timeout(
        "select count(*) as count from customer limit 1", db_path, max_len=10
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Edge case queries""")
    return


@app.cell
def _(Execption, db_path, dbman):
    def _():
        edge_cases = [
            ("Empty result", "select * from customer where 1=0"),
            ("Invalid table", "select * from nonexistence_table"),
            ("Malformed sql", "seleect * form customer"),
            (
                "SQL injection attemp",
                "select * from customer; drop table customer",
            ),
        ]

        for test_name, query in edge_cases:
            try:
                result = dbman._exec_query_sqlite_timeout(
                    query, db_path, max_len=5
                )
                if result is None:
                    print(f"[x] {test_name} properly handled (returned None)")
                else:
                    print(f"[!] {test_name} returned data - shape {result.shape}")
            except Execption as e:
                print(f"[x] {test_name}: Exeption caught - {type(e).__name__}")


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Timeout testing""")
    return


@app.cell
def _(db_path, dbman, time):
    def _():
        slow_query = """
        with recursive slow AS (
            select 1 as n
            union all
            select n + 1 from slow where n < 5000000
        )
        select count(*) from slow
        """

        start_time = time.time()
        result = dbman._exec_query_sqlite_timeout(slow_query, db_path, timeout=2)
        elapsed = time.time() - start_time

        if result is None and elapsed < 5:
            print("[x] Timeout works!")
        else:
            print(
                f"[?] Timeout test result: {elapsed:.2f}s, result={'None' if result is None else 'Data'} "
            )


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Testing row limiting""")
    return


@app.cell
def _(db_path, dbman):
    def _():
        large_table_query = "select * from lineitem"

        limits = [5, 100, 1000]
        for limit in limits:
            result = dbman._exec_query_sqlite_timeout(
                large_table_query, db_path, max_len=limit
            )
            if result is not None:
                actual_rows = len(result)
                expected_rows = min(limit, len(result))

                assert actual_rows <= limit

                print(f"[x] Got {actual_rows} (<= {limit})")
            else:
                print("f[!] Query returned None")


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Test multiple databases support""")
    return


@app.cell
def _(dbman):
    def _():
        print(f"Active connections: {list(dbman.conns.keys())}")
        print(f"Connection count: {len(dbman.conns)}")


        print(f"\nAdding database connection\n")
        chinook_db_path = "data/chinook.db" 
        dbman.start_sqlite(chinook_db_path)

        print(f"Active connections: {list(dbman.conns.keys())}")
        print(f"Connection count: {len(dbman.conns)}")

        for path, conn in dbman.conns.items():
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                print(f"[x] Connection {path}: Active")
            except Exception as e:
                print(f"[x] Connection {path}: Error - {e}")

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Test cleanup""")
    return


@app.cell
def _(db_path, dbman):
    def _():
        initial_conn_count = len(dbman.conns)
        print(f"Before cleanup: {initial_conn_count} connections")

        dbman.close_all()
        assert len(dbman.conns) == 0

        print("[x] All connections closed")
        print(f"After cleanup: {len(dbman.conns)} connections")

        # Test using database after close (should fail gracefully)
        try:
            result = dbman.get_schema(db_path)
            print("[!] Should have failed after close_all")
        except Exception as e:
            print(f"[x] Properly failed after close: {type(e).__name__}")


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Test data type""")
    return


@app.cell
def _(db_path, dbman, pd):
    def _():
        dbman.start_sqlite(db_path)

        # Test data type in result

        result = dbman._exec_query_sqlite_timeout("SELECT * FROM customer LIMIT 5", db_path)
        if result is not None:
          print("✓ Query result is DataFrame:", isinstance(result, pd.DataFrame))
          print(f"✓ Result shape: {result.shape}")
          print(f"✓ Column types: {dict(result.dtypes)}")
          print("✓ Sample data:")
          print(result.head(2))


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Schema validation""")
    return


@app.cell
def _(db_path, dbman):

    def _():
        # Detailed schema validation
        table_names, schemas = dbman.get_schema(db_path, add_sample_rows=True)
    
        for schema in schemas[:3]:  # Test first 3 tables
            table_name = schema["table_fullname"]
            print(f"\n--- Testing {table_name} ---")
    
            # Validate all expected keys exist
            required_keys = ["table_fullname", "column_names", "column_types", "sample_rows"]
            for key in required_keys:
                assert key in schema, f"Missing key {key} in {table_name}"
    
            # Validate columns match between names and types
            assert len(schema["column_names"]) == len(schema["column_types"])
            print(f"[x] Columns: {len(schema['column_names'])}")
    
            # Test actual query against this table
            test_query = f"SELECT {', '.join(schema['column_names'][:3])} FROM {table_name} LIMIT 2"
            result = dbman._exec_query_sqlite_timeout(test_query, db_path)
            if result is not None:
                print(f"[x] Query test passed: {result.shape}")
            else:
                print(f"[x] Query test failed")
    
        print("\n[x] All schema validation tests completed")


    _()
    return


if __name__ == "__main__":
    app.run()
