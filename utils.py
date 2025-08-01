import sqlglot
import io
import base64
import gzip
import pandas as pd


def extract_all_blocks(content: str, code_format: str | None = None) -> list:
    """Extract and return a list of sql blocks"""
    sql_blocks = []
    start = 0

    if code_format is None:
        code_format = ""

    while True:
        sql_query_start = content.find(f"```{code_format}", start)
        if sql_query_start == -1:
            break

        sql_query_end = content.find("```", sql_query_start + len(f"```{code_format}"))
        if sql_query_end == -1:
            break

        sql_block = content[
            sql_query_start + len(f"```{code_format}") : sql_query_end
        ].strip()

        sql_blocks.append(sql_block)

        start = sql_query_end + len("```")

    return sql_blocks


def split_sql(sql_s: str) -> list[str]:
    return [str(stmt) for stmt in sqlglot.parse(sql_s)]


def get_sqlite_path(db_folder: str, db_id: str):
    db_id = db_id.lower().replace("-", "_")
    return f"{db_folder}/{db_id}.sqlite"


def seripress_df(df, name="df"):
    """Serialize DataFrame with data type preservation"""
    data_dict = {}

    buff = io.StringIO()
    df.to_csv(buff, index=False)

    compressed = gzip.compress(buff.getvalue().encode("utf-8"))
    encoded = base64.b64encode(compressed).decode("utf-8")

    data_dict[f"{name}_csv_data"] = encoded

    # save datatypes metadata
    data_dict[f"{name}_dtypes"] = df.dtypes.astype(str).to_dict()

    # save index info if not a simple sequence
    if df.index.name or not df.index.equals(pd.RangeIndex(len(df))):
        data_dict[f"{name}_index"] = {"data": df.index.tolist(), "name": df.index.name}

    return data_dict


def deseripress_df(data_dict, name="df"):
    """Deserialize DataFrame with data type restoration"""

    encoded = data_dict[f"{name}_csv_data"]
    compressed = base64.b64decode(encoded.encode("utf-8"))
    csv_string = gzip.decompress(compressed).decode("utf-8")
    df = pd.read_csv(io.StringIO(csv_string))

    if f"{name}_dtypes" in data_dict:
        dtypes = data_dict[f"{name}_dtypes"]
        for col, dtype in dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except:
                    pass

    if f"{name}_index" in data_dict:
        index_info = data_dict[f"{name}_index"]
        df.index = pd.Index(index_info["data"], name=index_info["name"])

    return df


def hard_cut(
    str_e,
    length,
):
    if length:
        if len(str_e) > length:
            str_e = str_e[: int(length)] + "\n"
    return str_e


def get_values_from_table(csv_data_str):
    return "\n".join(csv_data_str.split("\n")[1:])
