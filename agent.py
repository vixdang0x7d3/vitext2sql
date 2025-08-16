from bitarray import bitarray

import pandas as pd

from database_manager import (
    DatabaseManager,
    SQLError,
)

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
        db_path: str,  # path to database folder
        db_id: str,  # the name of the database file, without .sqlite :-)
        db_manager: DatabaseManager,
        client: LLMClient,
        max_try: int = 3,
        max_iter: int = 10,
        early_stop: int = 4,
    ):
        self.max_try = 3
        self.max_iter = 10
        self.early_stop = 4
        self.sqlite_path = get_sqlite_path(db_path, db_id)

        self.db_manager = db_manager
        self.llm = client

    def self_refine(self, question, base_prompt, log_callback=None):
        self_refine_prompt = initial_prompt.format(
            question=question,
            base_prompt=base_prompt,
            db_type="sqlite",
        )

        iter = 0
        error_rec = bitarray()

        consistency_state = ConsistencyState.create_empty()

        final_sql = None
        final_result = None
        logs = []

        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                logs.append(msg)

        while iter < self.max_iter:
            response_sql = None
            max_try = self.max_try
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

            log(f"Iteration {iter + 1}: Generated SQL:\n{response_sql}")

            print(f"Try to run SQL in self-refine - Iteration {iter + 1}")
            print(f"SQL:\n{response_sql}")

            executed_result = self.db_manager.exec_query_sqlite(
                response_sql,
                self.sqlite_path,
                save_path=None,
            )

            if executed_result.success:
                log(
                    f"Executed successfully, rows affected: {executed_result.rows_affected}"
                )
            else:
                log(
                    f"Execution failed: {executed_result.error_type} - {executed_result.error_message}"
                )

            if not executed_result.success:
                error_rec.append(1)
            else:
                error_rec.append(0)

            # Early stop check for repeating failure
            if len(error_rec) > self.early_stop:
                last_four_errors = error_rec[:-4]
                if all(last_four_errors):  # all bits are 1
                    print("Repetitive execution failure, stopping")
                    break

            if executed_result.success:
                # Compressed query result for more efficient processing
                compressed_data = seripress_df(executed_result.data)

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
                unrecoverable_errors = [
                    SQLError.DATABASE_LOCKED,
                    SQLError.PERMISSION_DENIED,
                    SQLError.CONNECTION_ERROR,
                    SQLError.EXECUTION_ERROR,
                    SQLError.SAVE_WARNING,
                    SQLError.UNKNOWN_ERROR,
                ]

                if executed_result.error_type in unrecoverable_errors:
                    raise RuntimeError(
                        f"Unrecoverable database error: {executed_result.error_type} - {executed_result.error_message}"
                    )

                if executed_result.rows_affected == 0:
                    self_refine_prompt = correction_prompt.format(
                        sql=response_sql,
                        error_type="empty_result",
                        error_msg=(
                            "No row returned from query, try relaxing the filters and see if it works. "
                            "e.g. remove LIMIT, soften WHERE clauses"
                        ),
                    )

                else:
                    self_refine_prompt = correction_prompt.format(
                        sql=response_sql,
                        error_type=executed_result.error_type,
                        error_msg=executed_result.error_message,
                    )

            iter += 1

        # Print statistics
        print(f"Total iterations: {iter}")

        if len(error_rec) > 0:
            error_count = error_rec.count(1)
            successful = len(error_rec) - error_count

            print(
                f"Execution statistics: {successful} successes, {error_count} failures"
            )
            print(f"Success rate: {successful / len(error_rec) * 100:.1f}%")
            print(
                "Error pattern (last 10): "
                f"{error_rec[-10:].to01() if len(error_rec) >= 10 else error_rec.to01()}"
            )

        # notify user and log status
        success = iter < self.max_iter and final_result is not None

        return success, final_result, final_sql, "\n".join(logs)

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

    def health_check(self):
        """Simple health check for the text2sql agent"""
        print("=== Health Check ===")

        # Test database
        print("fix " + self.sqlite_path)
        try:
            result = self.db_manager.exec_query_sqlite(
                "SELECT 1", self.sqlite_path, save_path=None
            )
            print(
                f"✓ Database: {'OK' if result.success else 'FAILED: ' + result.error_message}"
            )
        except Exception as e:
            print(f"✗ Database: FAILED - {e}")

        # Test LLM
        # try:
        #     # Check API key for OpenAI clients
        #     if hasattr(self.llm, "health_check"):
        #         if self.llm.health_check():
        #             print("✓ LLM: OK")
        #         else:
        #             print("✗ LLM: FAILED - API connection failed")
        #     else:
        #         response = self.llm.get_model_response("write a dummy sql query", "sql")
        #         print(f"✓ LLM: {'OK' if response else 'FAILED'}")
        # except Exception as e:
        #     print(f"✗ LLM: FAILED - {e}")

        # Test schema
        try:
            tables = self.db_manager.exec_query_sqlite(
                "SELECT name FROM sqlite_master WHERE type='table'",
                self.sqlite_path,
                save_path=None,
            )
            print(
                f"✓ Schema: {'OK' if tables.success else 'FAILED: ' + result.error_message}"
            )
        except Exception as e:
            print(f"✗ Schema: FAILED - {e}")
