import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    import io
    import gzip
    import base64
    import json
    return base64, gzip, io, pd


@app.cell
def _(base64, gzip, io, pd):
    def seripress_df_with_dtypes_(df, name='df'):
        """Serialize DataFrame with data type preservation""" 
        data_dict = {}

        buff = io.StringIO()
        df.to_csv(buff, index=False)

        compressed = gzip.compress(buff.getvalue().encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('utf-8')

        data_dict[f'{name}_csv_data'] = encoded


        # save datatypes metadata 
        data_dict[f'{name}_dtypes'] = df.dtypes.astype(str).to_dict()


        # save index info if not a simple sequence
        if df.index.name or not df.index.equals(pd.RangeIndex(len(df))):
            data_dict[f'{name}_index'] = {
                'data': df.index.tolist(),
                'name': df.index.name
            }

        return data_dict


    def deseripress_df_with_dtypes_(data_dict, name='df'):
       """Deserialize DataFrame with data type restoration"""

       encoded = data_dict[f'{name}_csv_data']
       compressed = base64.b64decode(encoded.encode('utf-8'))     
       csv_string = gzip.decompress(compressed).decode('utf-8')
       df = pd.read_csv(io.StringIO(csv_string))

       if f'{name}_dtypes' in data_dict:
          dtypes = data_dict[f'{name}_dtypes']
          for col, dtype in dtypes.items():
              if col in df.columns:
                  try:
                      df[col] = df[col].astype(dtype)
                  except:
                      pass 

       if f'{name}_index' in data_dict:
           index_info = data_dict[f'{name}_index']
           df.index = pd.Index(index_info['data'], name=index_info['name'])

       return df
    return deseripress_df_with_dtypes_, seripress_df_with_dtypes_


@app.cell
def _(pd):
    df = pd.DataFrame({
        'A' : [1, 2, 3],
        'B' : ['x', 'y', 'z'],
        'C' : [1.1, 2.2, 3.3]
    })

    df
    return (df,)


@app.cell
def _(df, seripress_df_with_dtypes_):
    data = seripress_df_with_dtypes_(df, name='urmom')
    data
    return (data,)


@app.cell
def _(data, deseripress_df_with_dtypes_):
    restored_df = deseripress_df_with_dtypes_(data, name='urmom')
    restored_df
    return


@app.cell
def _():
    def get_prompt_dialect_basic():
        return '''
    ```sql
    SELECT DISTINCT "column_name" FROM "table_name" WHERE ... 
    ``` 
    (Replace "table_name" with the actual table name. Enclose table and column names with double quotations if they contain special characters or match reserved keywords.)
    '''


    decimal_place_prompt = "If the task description does not specify the number of decimal places, retain all decimals to four places.\n\n"


    def get_self_refine_prompt(
        db_schema,
        exemplars,
        query,
        format_csv,
        db_type,
    ):
        refine_prompt = db_schema + "\n\n"
    
        refine_prompt += (
            "Some few-shot examples after column exploration may be helpful:\n"
            + exemplars + "\n\n"
            if exemplars
            else ""
        )
    
        refine_prompt += (
            f"Follow the answer format like: {format_csv}.\n\n" if format_csv else ""
        )
    
        refine_prompt += (
            "Task: "
            + query
            + "\n"
            + f"\nPlease think step by step and answer only one complete SQL in {db_type} dialect in ```sql``` format.\n\n"
        )
    
        refine_prompt += f"SQL usage example:\n{get_prompt_dialect_basic()}\n"
    
        refine_prompt += "When asked something without stating name or id, return both of them. e.g. Which products ...? The answer should include product_name and product_id.\n"
    
        refine_prompt += "When asked percentage decrease, you should return a positive value. e.g. How many percentage points in 2021 decrease compared to ...? The answer should be a positive value indicating the decreased number. Try to use ABS().\n"
    
        refine_prompt += "If asked two tables, you should reply with the last one instead of combining two tables. e.g. Identifying the top five states ... examine the state that ranks fourth overall and identify its top five counties. You should only answer top five counties.\n"

        return refine_prompt
    return decimal_place_prompt, get_self_refine_prompt


@app.cell
def _(get_self_refine_prompt):
    print(get_self_refine_prompt(
        db_schema="[db_schema]", 
        exemplars="[exemplars]", 
        query="[query]", 
        format_csv="[format_csv]",
        db_type="sqlite"
    ))
    return


@app.cell
def _(decimal_place_prompt):
    def get_self_consistency_prompt(question, format_csv):
        self_consistency_prompt = f"Please check the answer again by reviewing task:\n{question}\n\nReview RELEVANT TABLES and COLUMNS and POSSIBLE CONDITIONS and then give the final SQL query. Don't output other queries. If you think the answer is right, just output the current SQL.\n"
        self_consistency_prompt += decimal_place_prompt
        self_consistency_prompt += (
            f"The answer format should be like: {format_csv}\n" if format_csv else ""
        )

        return self_consistency_prompt

    return (get_self_consistency_prompt,)


@app.cell
def _(get_self_consistency_prompt):
    print(get_self_consistency_prompt("[question]", "[format_csv]"))
    return


if __name__ == "__main__":
    app.run()
