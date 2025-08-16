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
 
        if csv_string.strip() == "":
              return pd.DataFrame()
 
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
def _(pd):
    df2 = pd.DataFrame()
    df2
    return (df2,)


@app.cell
def _(df2, seripress_df_with_dtypes_):
    empty_data = seripress_df_with_dtypes_(df2, name='empty_df')
    empty_data
    return (empty_data,)


@app.cell
def _(deseripress_df_with_dtypes_, empty_data):
    restored_empty_df = deseripress_df_with_dtypes_(empty_data, name='empty_df')
    return (restored_empty_df,)


@app.cell
def _(restored_empty_df):
    print(restored_empty_df)
    return


if __name__ == "__main__":
    app.run()
