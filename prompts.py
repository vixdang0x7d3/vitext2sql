correction_prompt = """Input SQL:
{sql}

Error type:
{error_type}

Error Message:
{error_msg}


Please correct the SQL based on the previous context and provide your thinking process.

Requirements:
- Output exactly one corrected SQL query
- Use the format: ```sql
-- Description: [brief explaination of the fix]
[Corrected SQL query]
"""

decimal_place_prompt = "If the task description does not specify the number of decimal places, retain all decimals to four places.\n\n"

initial_prompt = """Your task is to generate a SQL query that satisfy a user natural language question.

Please think step by step and answer only one complete SQL in {db_type} dialect in the following format:
```sql
[SQL Query]
```


User natural language question:
{question}

Revelent database entities and some additional information that may be useful to answer the question.
{base_prompt}
"""

consistency_prompt = """Please check the answer again by reviewing the question:
{question}

Instructions:
- Review RELEVANT TABLES and COLUMNS and POSSIBLE CONDITIONS and then give the final SQL query.
- Don't output other queries. 
- If you think the answer is right, just output the current SQL.
- If the current answer is "Empty", just output the current SQL.

If the task description does not specify the number of decimal places, retain all decimals to four places.

{csv_format}
"""

schema_linking_prompt = """You are doing table level schema linking. Given a table with schema information and the task,
you should think step by step and decide whether this table is related to the task.Some information may be in related tables.
Use foreign key relationships if necessary to determine relevance.
You should answer Y/N only. If the answer is Y, you should add columns that you think is related in python list format.

Please answer only in json code block like:
```json
{{
    "think": "think step by step to decide",
    "answer": "Y or N only",
    "columns": [col_name1, col_name2]
}}
```

Table info: {0}
Task: {1}
{2}
"""
