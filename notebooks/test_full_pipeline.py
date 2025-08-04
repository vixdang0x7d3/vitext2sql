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


    import pre.retrive_external_context
    from pre.sqlite import extract_ddl_to_csv, export_all_tables_to_json
    from pre.setup_vector_chromadb import VietnameseRAGSystem
    from pre.retrive_external_context import retrieve_from_collections, save_prompt_context
    from pre.reconstruct_data import compress_ddl
    from pre.schema_linking import ask_model_sl
    return (
        ask_model_sl,
        compress_ddl,
        export_all_tables_to_json,
        extract_ddl_to_csv,
        glob,
        os,
        retrieve_from_collections,
        save_prompt_context,
        time,
    )


@app.cell
def _(
    VietnameseRagSystem,
    export_all_tables_to_json,
    extract_ddl_to_csv,
    glob,
    os,
):
    question = "Liệt kê tất các cầu thủ c"
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
    return db_des, db_folder, db_name, question


@app.cell
def _(
    ask_model_sl,
    compress_ddl,
    db_des,
    db_folder,
    db_name,
    final_context_prompt_path,
    os,
    question,
    retrieve_from_collections,
    save_prompt_context,
    time,
):
    id = int(time.time())

    desc_exemplars = retrieve_from_collections(db_des, question, db_name)

    save_prompt_context(
        db_des=db_des, results=desc_exemplars, id=id, db_name=db_name
    )

    compress_ddl(
        db_name=db_name,
        id=id,
        add_description=True,
        add_sample_rows=True,
        rm_digits=True,
        schema_linked=False,
        clear_long_eg_des=True,
    )

    ask_model_sl(
        task=question,
        id=id,
        db_name=db_name,
    )

    final_context_promp_path = os.path.join(db_folder,  'final_context_prompt', str(id) + '.txt')

    prompt_template = ""
    with open(final_context_prompt_path, 'r') as f:
        prompt_template = f.read()

    return


if __name__ == "__main__":
    app.run()
