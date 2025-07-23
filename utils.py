def extract_all_blocks(content, code_format):
    """Extract and return a list of sql blocks"""
    sql_blocks = []
    start = 0

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
