import os
from bitarray import bitarray

import pandas as pd

from database_manager import DatabaseManager
from llm_client import LLMClient

from utils import (
    get_sqlite_path,
    seripress_df,
    deseripress_df,
)

from prompts import (
    initial_prompt,
    correction_prompt,
)

from result_validation import (
    ConsistencyState,
    process_result,
)


class Agent:
    def __init__(
        self,
        db_path: str,
        db_id: str,
        db_manager: DatabaseManager,
        client: LLMClient,
    ):
        self.max_try = 3
        self.max_iter = 10
        self.early_stop = True
        self.sqlite_path = get_sqlite_path(db_path, db_id)

        self.db_manager = db_manager
        self.llm = client

    def self_refine(
        self,
        question,
        base_prompt,
        csv_save_path,
        sql_save_path,
    ):
        self_refine_prompt = initial_prompt.format(
            base_prompt=base_prompt,
        )

        iter = 0
        error_rec = bitarray()
        consistency_state = ConsistencyState.create_empty()

        final_sql = None
        final_result = None

        while iter < self.max_iter:
            response_sql = None
            max_try = 3
            while max_try > 0:
                response = self.llm.get_model_response(self_refine_prompt, "sql")

                if isinstance(response, list) and len(response) == 1:
                    response_sql = response[0]
                    break
                else:
                    self_refine_prompt = "Output only one SQL only."
                    max_try -= 1

            if response_sql is None:
                print("Error generating SQL")
                break

            print(f"Try to run SQL in self-refine - Iteration {iter + 1}")
            print(f"SQL: {response_sql}")

            executed_result = self.db_manager.exec_query_sqlite(
                response,
                self.sqlite_path,
                save_path=None,
            )

            if not executed_result.success:
                error_rec.append(0)
            else:
                error_rec.append(1)

            # Early stop check for repeating failure
            if len(error_rec) > 3:
                last_four = error_rec[:-4]
                if not any(last_four):  # all bits are zeros
                    if os.path.exists(csv_save_path):
                        os.remove(csv_save_path)
                    print("Repetitive empty result, stopping")
                    break

            # Handle successful execution
            if executed_result is not None and executed_result.success:
                # Compressed query result for more efficient processing
                compressed_data = seripress_df(executed_result)

                # Validate result consistency,
                # construct response prompt based on validation result
                success, updated_prompt, updated_state = process_result(
                    compressed_data=compressed_data,
                    question=question,
                    response=response_sql,
                    db_type="sqlite",
                    consistency_state=consistency_state,
                )

                consistency_state = updated_state

                if success:
                    print("consistency results achieved")
                    final_sql = response_sql
                    final_result = compressed_data
                    break

                if updated_prompt:
                    self_refine_prompt = updated_prompt
            else:
                # Handle execution error
                error_message = (
                    executed_result.error_message
                    if executed_result
                    else "Unknown execution error"
                )

                self_refine_prompt = correction_prompt.format(
                    sql=response_sql,
                    error=error_message,
                )

            iter += 1

        # Print statistics
        print(f"Total iterations: {iter}")
        if len(error_rec) > 0:
            success_count = error_rec.count(1)
            failure_count = error_rec.count(0)

            print(
                f"Execution statistics: {success_count} successes, {failure_count} failures"
            )
            print(f"Success rate: {success_count / len(error_rec) * 100:.1f}%")
            print(
                "Error pattern (last 10): "
                f"{error_rec[-10:].to01() if len(error_rec) >= 10 else error_rec.to01()}"
            )

        success = iter < self.max_iter and final_result is not None

        return success, final_result, final_sql

    def get_result_dataframe(self, compressed_result: dict) -> pd.DataFrame:
        """Helper method to get DataFrame from compressed result"""
        return deseripress_df(compressed_result)

    def get_result_csv(self, compressed_result: dict) -> str:
        """Helper method to get CSV string from compressed result"""
        return compressed_result["df_csv_data"]

    def save_results(
        self, compressed_result: dict, sql_query: str, csv_path: str, sql_path: str
    ):
        """Helper method to save compressed results to disk if needed"""
        # Save CSV
        with open(csv_path, "w") as f:
            f.write(compressed_result["df_csv_data"])

        # Save SQL
        with open(sql_path, "w") as f:
            f.write(sql_query)
