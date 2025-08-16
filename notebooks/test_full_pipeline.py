import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import glob

    import time

    # add parent directory to kernel's path
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


    from database_manager import DatabaseManager
    from llm_client import  OpenAIClient, create_ollama_client
    from agent import Agent

    import pre.retrive_external_context
    from pre.sqlite import extract_ddl_to_csv, export_all_tables_to_json
    from pre.setup_vector_chromadb import VietnameseRAGSystem
    from pre.retrive_external_context import retrieve_from_collections, save_prompt_context
    from pre.reconstruct_data import compress_ddl
    from pre.schema_linking import ask_model_sl
    return (
        Agent,
        DatabaseManager,
        OpenAIClient,
        ask_model_sl,
        compress_ddl,
        export_all_tables_to_json,
        extract_ddl_to_csv,
        glob,
        mo,
        os,
        retrieve_from_collections,
        save_prompt_context,
        time,
    )


@app.cell
def _():
    import sqlalchemy
    DATABASE_URL = "sqlite:///pre/db/baseball_1/baseball_1.sqlite"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return (engine,)


@app.cell
def _(engine, mo):
    _df = mo.sql(
        f"""
        select name from sqlite_master where type='table'
        """,
        engine=engine
    )
    return


@app.cell
def _(engine, mo):
    _df = mo.sql(
        f"""
        pragma table_info('cầu_thủ')
        """,
        engine=engine
    )
    return


@app.cell
def _(
    VietnameseRagSystem,
    export_all_tables_to_json,
    extract_ddl_to_csv,
    glob,
    os,
):
    question = "Liệt kê tất các cầu thủ"
    db_des = True
    db_name = "baseball_1"

    db_folder = os.path.join('pre/db', db_name)
    db_path = os.path.join(db_folder, db_name + ".sqlite")


    if not os.path.exists(db_path):
        print(f"sqlite file not found: {db_path}")

    schema_path = os.path.join(db_folder, "schema")
    os.makedirs(schema_path, exist_ok=True)

    ddl_path = os.path.join(schema_path, "DDL.csv")
    if not os.path.exists(ddl_path):
        print(f"DDL.csv not found: {ddl_path} ")
        extract_ddl_to_csv(db_name, 'DDL.csv')

    json_files = glob.glob(os.path.join(schema_path, "*.json"))

    if json_files:
        pass
    else:
        print("No json in directory")
        result = export_all_tables_to_json(
            db_name,
            sample_limit=3,
        )

    vector_db_path = os.path.join(db_folder, "db_chroma")

    if not os.path.exists(vector_db_path):
        input_file = os.path.join(db_folder, "db_des", "db_des.txt")
        if os.path.exists(input_file):
            print("Generating vector index for databaser description")
            rag_system = VietnameseRagSystem()
            rag_system.setup_database(db_name)
        else:
            print("No database description")
            db_des=False
    return db_des, db_folder, db_name, db_path, question


@app.cell
def _(OpenAIClient):
    sl_client = OpenAIClient(
        model="o3-2025-04-16", 
        max_context_length=200_000,
        system_prompt="",
    )
    return (sl_client,)


@app.cell
def _(
    ask_model_sl,
    compress_ddl,
    db_des,
    db_folder,
    db_name,
    db_path,
    os,
    question,
    retrieve_from_collections,
    save_prompt_context,
    sl_client,
    time,
):
    id = int(time.time())

    desc_exemplars, logs = retrieve_from_collections(db_folder, db_path, db_des, question, db_name)

    save_prompt_context(
        db_des=db_des, 
        db_folder=db_folder, 
        db_path=db_path,
        results=desc_exemplars, 
        id=id, 
        db_name=db_name
    )

    compress_ddl(
        db_folder=db_folder,
        db_path=db_path,
        db_name=db_name,
        id=id,
        add_description=True,
        add_sample_rows=True,
        rm_digits=True,
        schema_linked=False,
        clear_long_eg_des=True,
    )

    ask_model_sl(
        db_folder,
        db_path,
        task=question,
        id=id,
        chat_session=sl_client,
        db_name=db_name,
    )

    final_context_prompt_path = os.path.join(db_folder,  'final_context_prompts', str(id) + '.txt')

    prompt_template = ""
    with open(final_context_prompt_path, 'r') as f:
        prompt_template = f.read()

    return (prompt_template,)


@app.cell
def _(prompt_template):
    print(prompt_template)
    return


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

    # response = client.get_model_response(
    #     prompt="""
    #     You're task is to generate a SQL query that satisfy a user natural language question.
    #     Please think step by step and answer only one complete SQL in sqlite dialect in the following format:
    #     ```sql
    #     [SQL Query]
    #     ```
    # 
    #     Here are information about the user's question and the database structure as well as some additional context:
    # 
    #     Database schema:
    #     cầu_thủ(họ, tên)
    # 
    #     Additional context:
    # 
    #     Question:
    #     Liệt kê họ tên các cầu thủ
    #     """,
    #     code_format="sql",
    # )
    # 
    # print(response[0])
    return (client,)


@app.cell
def _(DatabaseManager, db_path):
    dbman = DatabaseManager()
    dbman.start_sqlite(db_path)

    query_result = dbman.exec_query_sqlite(
        "SELECT * FROM `cầu_thủ`",
        db_path,
    )

    print(dbman.format_query_result(query_result))
    return (dbman,)


@app.cell
def _(Agent, client, dbman):
    agent = Agent(
        db_path="pre/db/baseball_1",
        db_id="baseball_1.sqlite",
        db_manager=dbman,
        client=client,
    )

    agent.health_check()
    return (agent,)


@app.cell
def _(question):
    print(f"Question: {question}")
    return


@app.cell
def _(agent, prompt_template, question):
    self_refine_result = agent.self_refine(
       question=question,
       base_prompt=prompt_template
    )
    return


if __name__ == "__main__":
    app.run()
