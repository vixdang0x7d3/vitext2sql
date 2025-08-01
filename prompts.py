correction_prompt = """Input SQL:
{sql}

Error Information:
{str(error)}


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

Here are information about the user's question and the database structure as well as some additional context:
{base_prompt}
"""

consistency_prompt = """Please check the answer again by reviewing the question:
{question}

Review RELEVANT TABLES and COLUMNS and POSSIBLE CONDITIONS and then give the final SQL query. Don't output other queries. If you think the answer is right, just output the current SQL.

If the task description does not specify the number of decimal places, retain all decimals to four places.

{csv_format}
"""
