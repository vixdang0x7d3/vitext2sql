import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys

    # add parent directory to kernel's path
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from database_manager import DatabaseManager
    from llm_client import  OpenAIClient, create_ollama_client
    from agent import Agent

    return Agent, DatabaseManager, OpenAIClient, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Loading and testing external resources""")
    return


@app.cell
def _():
    import sqlalchemy

    baseball_db_path = "data/testing/baseball_1.sqlite"

    baseball_db_url = f"sqlite:///{baseball_db_path}"
    baseball_db = sqlalchemy.create_engine(baseball_db_url)
    return baseball_db, baseball_db_path


@app.cell
def _(baseball_db, mo):
    _df = mo.sql(
        f"""
        SELECT name FROM sqlite_master WHERE type='table';
        """,
        engine=baseball_db
    )
    return


@app.cell
def _():
    from pprint import pprint
    import textwrap

    def print_wrapped(text, wrap_length=79):
        """
        New print_wrapped version that respect the
        indentations of the LLM output and the prompt
        """
        for line in text.splitlines():
            indent = len(line) - len(line.lstrip())
            wrapped = textwrap.fill(
                line,
                width=wrap_length,
                subsequent_indent=" " * indent,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            print(wrapped)

    question = (
        "liệt kê tên và họ của các cầu thủ đã chơi cho các đội có sân vận động ở California, "
        "và đã từng thắng World Series ít nhất một lần"
    )

    base_prompt = ""
    with open("data/testing/baseball_1_base_prompt.txt", "r") as f:
        base_prompt = f.read()
    print(f"Question: {question}\n")
    print_wrapped(base_prompt, wrap_length=120)
    return base_prompt, pprint, question


@app.cell
def _(baseball_db, mo):
    _df = mo.sql(
        f"""
        SELECT DISTINCT tsn.id_sân_vận_động, d.id_đội, c.id_cầu_thủ, c.họ || ' ' || c.tên AS họ_tên 
        FROM cầu_thủ AS c
        JOIN lần_xuất_hiện as lxh ON c.id_cầu_thủ = lxh.id_cầu_thủ
        JOIN đội as d ON lxh.id_đội = d.id_đội
        JOIN trận_đấu_sân_nhà   AS tsn ON tsn.id_đội    = d.id_đội
        JOIN sân_vận_động       AS svd ON svd.id_sân_vận_động = tsn.id_sân_vận_động
        """,
        engine=baseball_db
    )
    return


@app.cell
def _(baseball_db, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM sân_vận_động
        """,
        engine=baseball_db
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Agent setup""")
    return


@app.cell
def _(DatabaseManager, baseball_db_path):
    dbman = DatabaseManager()
    dbman.start_sqlite(baseball_db_path)

    result = dbman.exec_query_sqlite(
        "SELECT * FROM `cầu_thủ`",
        baseball_db_path,
    )

    print(dbman.format_query_result(result))
    return (dbman,)


@app.cell
def _(OpenAIClient):
    system_prompt = """You are a data science expert that can write excellent SQL queries. Below, you are provided with a database schema, a natural question, and some neccessary context. Your task is to understand the schema, the context, and generate a valid SQL query to answer the question.

    Instructions:
    - Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.

    - The generated query should return all of the information asked in the question without any missing or extra information.

    - Before generating the final SQL query, please think through the steps of how to write the query.

    - For string-matching scenarios, if the string is decided, don't use fuzzy query. e.g. Get the object's title contains the word "book". However, if the string is not decided, you may use fuzzy query and ignore upper or lower case. e.g. Get articles that mention "education".

    - If the task description does not specify the number of decimal places, retain all decimals to four places.

    - For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he’s to he%s)

    - When asked something without stating name or id, return both of them. e.g. Which products ...? The answer should include product_name and product_id.

    - When asked percentage decrease, you should return a positive value. e.g. How many percentage points in 2021 decrease compared to ...? The answer should be a positive value indicating the decresed number. Try to use ABS().

    - If asked two tables, you should reply with the last one instead of combining two tables. e.g. Identifying the top five states ... examine the state that ranks fourth overall and identify its top five counties. You should only answer top five counties.

    Take a deep breath and think step by step to find the correct SQL query.
    """

    # client = create_ollama_client(
    #    base_url="http://localhost:11434",
    #    model="qwen3:4b",
    #    temperature=0.8,
    #    system_prompt=system_prompt,
    # )

    client = OpenAIClient(
        model="o3-2025-04-16", 
        max_context_length=200_000,
        system_prompt=system_prompt,
    )

    response = client.get_model_response(
        prompt="""
        You're task is to generate a SQL query that satisfy a user natural language question.
        Please think step by step and answer only one complete SQL in sqlite dialect in the following format:
        ```sql
        [SQL Query]
        ```

        Here are information about the user's question and the database structure as well as some additional context:

        Database schema:
        cầu_thủ(họ, tên)

        Additional context:

        Question:
        Liệt kê họ tên các cầu thủ
        """,
        code_format="sql",
    )

    print(response[0])
    return (client,)


@app.cell
def _(Agent, client, dbman):
    agent = Agent(
        db_path="data/testing",
        db_id="baseball_1.sqlite",
        db_manager=dbman,
        client=client,
    )

    agent.health_check()
    return (agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Single run test""")
    return


@app.cell
def _(agent, base_prompt, question):
    print(question)

    success, final_result, final_sql = agent.self_refine(
        question=question, base_prompt=base_prompt
    )
    return final_result, final_sql, success


@app.cell
def _(agent, final_result, final_sql, success):
    if success:
        result_df = agent.get_result_dataframe(final_result)
        print(final_sql)
        print(result_df)

    return


@app.cell
def _(client, pprint):
    pprint(client.messages)
    return


@app.cell
def _(client):
    client.get_message_len()
    return


if __name__ == "__main__":
    app.run()
