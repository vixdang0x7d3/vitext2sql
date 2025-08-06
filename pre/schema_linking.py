from .utils import (
    search_file,
    get_api_name,
    get_dictionary,
    get_tb_info,
    get_external,
    compute_precision_recall,
    is_csv_empty,
    clear_name,
    remove_digits,
)


from llm_client import OpenAIClient  # ty: ignore

from .reconstruct_data import compress_ddl
import os
import json
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import re
import numpy as np
from itertools import islice
import time


def reduce_columns(sql: str, subset_columns: set[str]) -> str:
    # Bắt đúng tên bảng
    table_match = re.search(
        r"create\s+(?:or\s+replace\s+)?table\s+`([^`]+)`", sql, re.IGNORECASE
    )
    assert table_match, sql
    table_name = table_match.group(1)

    # Lấy block định nghĩa cột
    columns_block_match = re.search(
        r"\((.*?)\)\s*(PARTITION|CLUSTER|OPTIONS|;|$)", sql, re.DOTALL | re.IGNORECASE
    )
    if not columns_block_match:
        raise ValueError("Cannot extract columns block.")
    columns_block = columns_block_match.group(1)

    lines = columns_block.splitlines()
    filtered_lines = []
    for line in lines:
        line = line.strip().rstrip(",")
        if not line or line.upper().startswith("FOREIGN KEY"):
            continue

        # Dùng regex để lấy tên cột trong backtick
        col_match = re.match(r"`([^`]+)`", line)
        if not col_match:
            continue
        col_name = col_match.group(1)

        if col_name in subset_columns:
            filtered_lines.append(f"  {line},")

    if filtered_lines:
        filtered_lines[-1] = filtered_lines[-1].rstrip(",")

    new_sql = f"CREATE TABLE `{table_name}` (\n" + "\n".join(filtered_lines) + "\n);"
    return new_sql


def reduce_ddl(linked_json="", reduce_col=False, db_name="", id=1):
    print("Doing schema linking")

    with open(linked_json, encoding="utf-8") as f:
        tbs = json.load(f)

    table_names = []
    columns = {}

    for tb in tbs:
        if "answer" in tb:
            if tb["answer"] == "Y":
                table_names.append(tb["table name"])
                columns[tb["table name"]] = tb["columns"]
        else:
            raise NotImplementedError
            print(tb)
            table_names.append(tb)

    if not table_names:
        print("Empty result in table_names for this question")

    print("Doing sl for")
    table_names_no_digit = [remove_digits(i) for i in table_names]

    db_folder = os.path.join("pre/db", db_name)
    output_csv = os.path.join(db_folder, "schema", "DDL.csv")

    temp_file = os.path.join(db_folder, "ddl_sl")
    os.makedirs(temp_file, exist_ok=True)

    with (
        open(output_csv, "r", newline="", encoding="utf-8", errors="ignore") as infile,
        open(
            os.path.join(temp_file, str(id) + ".csv"),
            "w",
            newline="",
            encoding="utf-8",
            errors="ignore",
        ) as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)
        row_count = 0
        row_count_rm = 0
        total_count = 0
        row_list_all = []
        row_list = []
        for row in reader:
            assert row[-1].upper().startswith("CREATE"), row
            if "." in row[0]:
                row[0] = row[0].split(".")[-1]

            json_pth = output_csv.replace("DDL.csv", row[0].strip() + ".json")
            if os.path.exists(json_pth):
                with open(json_pth, encoding="utf-8") as f:
                    table_fullname = json.load(f)["table_fullname"]
            else:
                print(f" {json_pth} doesn't exist")
                continue

            if any(
                remove_digits(table_fullname) in item for item in table_names_no_digit
            ):
                row_count_rm += 1
                row_list_all.append(row)

            if any(table_fullname == item for item in table_names):
                row_count += 1

                if reduce_col:
                    assert table_fullname in columns, print(table_names, table_fullname)
                    # print(table_fullname)
                    # print(columns[table_fullname])
                    # print(row[-1])
                    row[-1] = reduce_columns(row[-1], columns[table_fullname])
                    print("After", row)
                row_list.append(row)

            total_count += 1
        print(
            f" tables before linking: {total_count}, tables after linking: {row_count}, tables rm digits after linking: {row_count_rm}"
        )
        if 0 < row_count < 10 or row_count_rm > 1000 or reduce_col:
            writer.writerows(row_list)
        elif row_count_rm:
            print("RM digits", len(row_list))
            writer.writerows(row_list_all)

    compress_ddl(
        id=id,
        db_name=db_name,
        add_description=True,
        add_sample_rows=True,
        rm_digits=True,
        schema_linked=True,
        clear_long_eg_des=True,
        reduce_col=reduce_col,
    )


def ask_model_sl(task, id, db_name):
    def process_example(task, tb_info, db_name):
        # if ex_id.startswith("local"):
        #     pass

        # task_path = r"D:\RAG_Vitext2sql\task.txt"
        # with open(task_path,encoding='utf-8') as f:
        #     task = f.read()

        chat_session = OpenAIClient(model="gpt-4.1", max_context_length=200_000)
        result = ask_model_sl_(tb_info, task, chat_session, db_name, id=id)
        return result

    db_folder = os.path.join("pre/db", db_name)
    output_path = os.path.join(db_folder, "prompts")

    # assert len(tb_info_pth) == 1
    with open(os.path.join(output_path, str(id) + ".txt"), encoding="utf-8") as f:
        tb_info = f.read()
    if len(tb_info) > 20000:
        # linked_dic = {}
        print("Doing table-level schema linking")
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(
                    process_example, task=task, tb_info=tb_info, db_name=db_name
                )
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing"
            ):
                result = future.result()
            #     if ex_id is not None:
            #         linked_dic[ex_id] = result
            output_path = os.path.join(db_folder, "sl_response")
            os.makedirs(output_path, exist_ok=True)
            with open(
                os.path.join(output_path, str(id) + ".txt"), "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(result, indent=4, ensure_ascii=False))
            reduce_ddl(
                linked_json=os.path.join(output_path, str(id) + ".txt"),
                reduce_col=True,
                db_name=db_name,
                id=id,
            )
    else:
        output_path = os.path.join(db_folder, "final_context_prompts")
        os.makedirs(output_path, exist_ok=True)

        with open(
            os.path.join(output_path, str(id) + ".txt"), "w", encoding="utf-8"
        ) as f:
            external_knowledge = os.path.join(db_folder, "external_knowledge")

            output_sql_ex_file = os.path.join(external_knowledge, "sql_ex_context")

            output_db_des_file = os.path.join(external_knowledge, "db_des_context")

            with open(
                os.path.join(output_sql_ex_file, str(id) + ".txt"), encoding="utf-8"
            ) as a:
                external_knowledge = a.read()
            tb_info += (
                f"External knowledge that might be helpful: \n{external_knowledge}\n"
            )
            with open(
                os.path.join(output_db_des_file, str(id) + ".txt"), encoding="utf-8"
            ) as a:
                external_knowledge = a.read()
            tb_info += f"\n{external_knowledge}\n"
            f.writelines(tb_info)


ask_prompt = """
You are doing table level schema linking. Given tables with schema information and the task,
you should think step by step and decide whether this table is related to the task.Some information may be in related tables not just in these tables.
Use foreign key relationships if necessary to determine relevance.
You should answer Y/N only. If the answer is Y, you should add columns that you think is related in python list format.

Return **a JSON array of exactly 3 elements**, each inside a JSON code block, like this:

```json
  {{
    "think": "step by step",
    "answer": "Y or N",
    "columns": ["col1","col2"],
    "table name": "table_name"
  }},
  {{
    "think": "step by step",
    "answer": "Y or N",
    "columns": ["col1","col2"],
    "table name": "table_name"
  }},
  {{
    "think": "step by step",
    "answer": "Y or N",
    "columns": ["col1","col2"],
    "table name": "table_name"
  }}
```

Table info: {0}
Task: {1}
External Knowledge:
{2}
"""

# def ask_model_sl_(tb_info, task, chat_session,db_name,id):
#     tbs = get_tb_info(tb_info)
#     # external = get_external(tb_info)
#     linked = []
#     db_folder = os.path.join("db", db_name)
#     external_knowledge =  os.path.join(db_folder, "external_knowledge")

#     output_db_des_file = os.path.join(external_knowledge, "db_des_context")

#     with open(os.path.join(output_db_des_file, str(id)+".txt"), encoding="utf-8") as a:
#         external_knowledge = a.read()
#     db_des = f"External knowledge that might be helpful: \n{external_knowledge}\n"

#     def chunk_list(lst, n):
#         """Chia list thành các chunk n phần tử"""
#         for i in range(0, len(lst), n):
#             yield lst[i:i+n]

#     linked = []
#     for chunk in chunk_list(tbs, 3):  # Mỗi lần 3 bảng
#         chat_session.init_messages()
#         max_try = 3

#         # Ghép 3 bảng thành 1 input
#         tb_text = "\n\n".join(chunk)
#         input = ask_prompt.format(tb_text, task, db_des)
#         while max_try:
#             response = chat_session.get_model_response(input, "json")

#             print(response)
#             for item in response:
#                 try:
#                     data = json.loads(item)
#                     assert data["answer"] in ["Y", "N"], 'data["answer"] should be in ["Y", "N"]'
#                     data["table name"] = re.search(r'^Table full name:\s*(.+)$', tb, re.MULTILINE).group(1)
#                     break
#                 except Exception as e:
#                     input = e+"Please generate again."
#             max_try -= 1
#         if max_try == 0:
#             print("Failed", re.search(r'^Table full name:\s*(.+)$', tb, re.MULTILINE).group(1))
#             continue
#         # print(data)
#         linked.append(data)


#     return linked
def ask_model_sl_(tb_info, task, chat_session, db_name, id):
    tbs = get_tb_info(tb_info)
    linked = []

    # Đọc external knowledge
    db_folder = os.path.join("pre/db", db_name)
    external_knowledge_path = os.path.join(db_folder, "external_knowledge")

    db_des_file = os.path.join(external_knowledge_path, "db_des_context", f"{id}.txt")
    if os.path.exists(db_des_file):
        with open(db_des_file, encoding="utf-8") as f:
            external_knowledge = f.read()
        db_des = f"External knowledge that might be helpful: \n{external_knowledge}\n"
    else:
        db_des = ""

    # Hàm chia list thành các chunk n phần tử
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for chunk in chunk_list(tbs, 3):  # mỗi lần 3 bảng
        chat_session.init_messages()
        max_try = 2
        tb_text = "\n\n".join(chunk)
        input_prompt = ask_prompt.format(tb_text, task, db_des)

        success = False
        while max_try:
            response = chat_session.get_model_response(input_prompt, "json")
            time.sleep(0.5)
            print(response)  # debug
            if len(response) == 1:
                print("ngu")
            try:
                # response là list JSON string -> parse từng cái
                for item, tb in zip(response, chunk):
                    data = json.loads(item)
                    assert data["answer"] in ["Y", "N"], (
                        'data["answer"] should be in ["Y", "N"]'
                    )
                    # table_name = re.search(r'^Table full name:\s*(.+)$', tb, re.MULTILINE).group(1)
                    # data["table name"] = table_name
                    linked.append(data)
                success = True
                break  # thoát vòng while nếu thành công
            except Exception as e:
                max_try -= 1
                input_prompt = f"{str(e)}. Please generate again."

        if not success:
            for tb in chunk:
                table_name = re.search(
                    r"^Table full name:\s*(.+)$", tb, re.MULTILINE
                ).group(1)
                print("Failed", table_name)

    return linked


# ask_model_sl()
# reduce_ddl(reduce_col=True)
