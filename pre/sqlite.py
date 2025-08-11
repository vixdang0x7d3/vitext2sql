import sqlite3
import os
import json
from datetime import datetime
import random
import csv


def create_database(db_name="baseball_1"):
    """
    Tạo database SQLite với schema tội phạm và thêm dữ liệu mẫu
    """
    # Kết nối đến database (tự động tạo file nếu chưa tồn tại)

    db_path = "db/" + db_name
    os.makedirs("db", exist_ok=True)
    connection = sqlite3.connect(db_path + ".sqlite")
    cursor = connection.cursor()

    # cursor.execute("DROP TABLE IF EXISTS `nhượng_quyền_thương_mại_của_các_đội`")
    # cursor.execute("DROP TABLE IF EXISTS `sân_vận_động`")
    # cursor.execute("DROP TABLE IF EXISTS `trường_đại_học`")
    # cursor.execute("DROP TABLE IF EXISTS `cầu_thủ`")
    # cursor.execute("DROP TABLE IF EXISTS `đội`")
    # cursor.execute("DROP TABLE IF EXISTS `giải_đấu_sau_mùa_giải`")
    # cursor.execute("DROP TABLE IF EXISTS `bình_chọn_giải_thưởng_dành_cho_huấn_luận_viên`")
    # cursor.execute("DROP TABLE IF EXISTS `bình_chọn_giải_thưởng_dành_cho_cầu_thủ`")
    # cursor.execute("DROP TABLE IF EXISTS `giải_thưởng_dành_cho_huấn_luyện_viên`")
    # cursor.execute("DROP TABLE IF EXISTS `giải_thưởng_dành_cho_cầu_thủ`")
    # cursor.execute("DROP TABLE IF EXISTS `giải_đấu_của_tất_cả_các_ngôi_sao`")
    # cursor.execute("DROP TABLE IF EXISTS `cầu_thủ_của_trường_đại_học`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_đánh_bóng`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_ném_bóng`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_phòng_ngự`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_phòng_ngự_sân_ngoài`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_huấn_luyện_viên`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_huấn_luyện_viên_theo_hiệp`")
    # cursor.execute("DROP TABLE IF EXISTS `lương`")
    # cursor.execute("DROP TABLE IF EXISTS `đại_lộ_danh_vọng`")
    # cursor.execute("DROP TABLE IF EXISTS `trận_đấu_sân_nhà`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_đội_theo_hiệp`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_đánh_bóng_sau_mùa_giải`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_ném_bóng_sau_mùa_giải`")
    # cursor.execute("DROP TABLE IF EXISTS `thành_tích_phòng_ngự_sau_mùa_giải`")
    # cursor.execute("DROP TABLE IF EXISTS `lần_xuất_hiện`")

    with open(db_path + "/create_tables.sql", "r", encoding="utf-8") as f:
        sql_script = f.read()

    #  Dùng executescript để chạy tất cả CREATE TABLE cùng lúc
    cursor.executescript(sql_script)

    # Commit và đóng kết nối
    connection.commit()
    connection.close()

    print(f"Database đã được tạo thành công tại: {db_path}")
    return db_path


def get_sqlite_data(
    path,
    add_description=False,
    add_sample_rows=False,
    gold_table_names=None,
    gold_column_names=None,
):
    """
    Hàm lấy thông tin từ SQLite database dưới dạng mô tả văn bản.
    """
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    prompts = ""
    global_foreign_key_map = []  # (from_table, from_col, to_table, to_col)
    all_table_foreign_keys = {}  # map table name -> list of its foreign keys

    # Bước 1: thu thập global foreign key mapping
    for table_name in tables:
        cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
        foreign_keys_info = cursor.fetchall()
        all_table_foreign_keys[table_name] = foreign_keys_info

        for fk in foreign_keys_info:
            from_col = fk[3]
            to_table = fk[2]
            to_col = fk[4]
            global_foreign_key_map.append((table_name, from_col, to_table, to_col))

    # Bước 2: tạo prompt cho từng bảng
    for table_name in tables:
        prompts += "\n" + "-" * 50 + "\n"
        prompts += f"Table full name: {table_name}\n"

        # Lấy thông tin cột
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns_info = cursor.fetchall()

        column_names = [col[1] for col in columns_info]
        column_types = [col[2] for col in columns_info]
        primary_keys = {col[1] for col in columns_info if col[5] == 1}

        # Lấy thông tin khóa ngoại của bảng này
        foreign_keys_info = all_table_foreign_keys[table_name]
        foreign_keys = {fk[3]: (fk[2], fk[4]) for fk in foreign_keys_info}

        # In thông tin từng cột
        for col_name, col_type in zip(column_names, column_types):
            col_desc = ""
            if col_name in primary_keys:
                col_desc += " [PRIMARY KEY]"
            if col_name in foreign_keys:
                to_table, to_col = foreign_keys[col_name]
                col_desc += f" [FOREIGN KEY → {to_table}.{to_col}]"
            prompts += f"Column name: {col_name} Type: {col_type}{col_desc}\n"

        # In danh sách foreign keys (nếu có)
        if foreign_keys:
            prompts += "Foreign keys:\n"
            for from_col, (to_table, to_col) in foreign_keys.items():
                prompts += f"    {from_col} → {to_table}.{to_col}\n"

        # In danh sách is_referenced_by (bảng khác trỏ đến bảng này)
        referenced_by = [
            (from_table, from_col)
            for (from_table, from_col, to_table, to_col) in global_foreign_key_map
            if to_table == table_name
        ]
        if referenced_by:
            prompts += "Is referenced by:\n"
            for from_table, from_col in referenced_by:
                prompts += f"    {from_table}.{from_col}\n"

        # In sample rows
        if add_sample_rows:
            try:
                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
                rows = cursor.fetchall()
                prompts += "Sample rows:\n"
                for row in rows:
                    prompts += f"    {row}\n"
            except Exception as e:
                prompts += f"Sample rows: Lỗi truy vấn - {e}\n"

    connection.close()
    return tables, prompts


def verify_database(db_path):
    """
    Kiểm tra và hiển thị thông tin database
    """
    print(f"\n=== KIỂM TRA DATABASE: {db_path} ===")

    # Sử dụng hàm get_sqlite_data để hiển thị thông tin
    table_names, prompts = get_sqlite_data(db_path, "test", add_sample_rows=True)
    print(f"Tables found: {table_names}")
    print(prompts)


def extract_ddl_to_csv(db_folder,db_path,db_name, output_csv="DDL.csv"):
    """
    Trích xuất DDL từ SQLite database và xuất ra file CSV

    Args:
        db_path (str): Đường dẫn đến file SQLite database
        output_csv (str): Tên file CSV output (mặc định: 'DDL.csv')
    """
    # db_folder = os.path.join("pre/db", db_name)
    # db_path = os.path.join(db_folder, db_name + ".sqlite")

    if not os.path.exists(db_path):
        print(f"File database không tồn tại: {db_path}")
    try:
        # Kết nối đến SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Lấy danh sách tất cả các bảng (không bao gồm bảng hệ thống)
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        tables = cursor.fetchall()

        # Tạo danh sách để lưu dữ liệu DDL
        ddl_data = []

        for table in tables:
            table_name = table[0]

            # Lấy DDL cho từng bảng
            cursor.execute(
                """
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name=?
            """,
                (table_name,),
            )

            ddl_result = cursor.fetchone()

            if ddl_result and ddl_result[0]:
                # Làm sạch và format DDL
                ddl = ddl_result[0].strip()

                # Thêm dấu chấm phẩy nếu chưa có
                if not ddl.endswith(";"):
                    ddl += ";"

                # Thêm vào danh sách
                ddl_data.append([table_name, ddl])

        # Ghi ra file CSV
        output_csv = os.path.join(db_folder, "schema", "DDL.csv")
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Ghi header
            writer.writerow(["table_name", "DDL"])

            # Ghi dữ liệu
            writer.writerows(ddl_data)

        conn.close()

        print(f"Đã xuất DDL thành công ra file: {output_csv}")
        print(f"Tổng số bảng được xuất: {len(ddl_data)}")

        # Hiển thị preview
        if ddl_data:
            print("\n Preview:")
            for i, (table_name, ddl) in enumerate(ddl_data[:3]):  # Hiển thị 3 bảng đầu
                print(f"  {i + 1}. {table_name}")
                print(f"     {ddl[:100]}{'...' if len(ddl) > 100 else ''}")

            if len(ddl_data) > 3:
                print(f"  ... và {len(ddl_data) - 3} bảng khác")

        return True

    except sqlite3.Error as e:
        print(f"Lỗi SQLite: {e}")
        return False
    except Exception as e:
        print(f"Lỗi: {e}")
        return False


def get_column_info(cursor, table_name):
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    columns_info = cursor.fetchall()
    column_names = [col[1] for col in columns_info]
    column_types = [col[2] for col in columns_info]
    primary_keys = [col[1] for col in columns_info if col[5] == 1]
    return column_names, column_types, primary_keys


def get_foreign_keys(cursor, table_name):
    cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
    fk_info = cursor.fetchall()
    foreign_keys = []
    for fk in fk_info:
        foreign_keys.append(
            {"from_column": fk[3], "to_table": fk[2], "to_column": fk[4]}
        )
    return foreign_keys


def get_sample_rows(cursor, table_name, limit=5):
    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}')
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    return [{col: row[i] for i, col in enumerate(column_names)} for row in rows]


def build_global_fk_map(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    fk_map = []
    for tbl in tables:
        cursor.execute(f'PRAGMA foreign_key_list("{tbl}")')
        for fk in cursor.fetchall():
            fk_map.append(
                {
                    "from_table": tbl,
                    "from_column": fk[3],
                    "to_table": fk[2],
                    "to_column": fk[4],
                }
            )
    return fk_map


def export_table_to_json(db_folder,db_path,db_name, table_name, sample_limit=5):
    try:
        # db_folder = os.path.join("db", db_name)
        # db_path = os.path.join(db_folder, db_name + ".sqlite")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Lấy thông tin cột, khóa chính, khóa ngoại
        column_names, column_types, primary_keys = get_column_info(cursor, table_name)
        foreign_keys = get_foreign_keys(cursor, table_name)
        sample_rows = get_sample_rows(cursor, table_name, sample_limit)
        global_fk_map = build_global_fk_map(cursor)

        # Tìm bảng nào đang tham chiếu tới bảng hiện tại
        is_referenced_by = []
        for fk in global_fk_map:
            if fk["to_table"] == table_name:
                is_referenced_by.append(
                    {"from_table": fk["from_table"], "from_column": fk["from_column"]}
                )

        # Tạo JSON
        table_data = {
            "table_name": table_name,
            "table_fullname": table_name,
            "column_names": column_names,
            "column_types": column_types,
            "description": [""] * len(column_names),
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "is_referenced_by": is_referenced_by,
            "sample_rows": sample_rows,
        }

        # Ghi JSON ra file
        schema_path = os.path.join(db_folder, "schema")
        output_file = os.path.join(schema_path, f"{table_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(table_data, f, indent=4, ensure_ascii=False)

        conn.close()
        return True

    except Exception as e:
        print(f" Lỗi khi xuất bảng {table_name}: {e}")
        return False


def export_all_tables_to_json(db_folder,db_path,db_name, sample_limit=5, exclude_tables=None):
    """
    Xuất tất cả các bảng trong database thành các file JSON riêng biệt

    Args:
        db_path (str): Đường dẫn đến SQLite database
        output_dir (str): Thư mục lưu các file JSON
        sample_limit (int): Số dòng mẫu cho mỗi bảng
        exclude_tables (list): Danh sách bảng cần loại trừ

    Returns:
        dict: Kết quả xuất file (success/failed tables)
    """
    if exclude_tables is None:
        exclude_tables = []

    try:
        # db_folder = os.path.join("db", db_name)
        # db_path = os.path.join(db_folder, db_name + ".sqlite")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Lấy danh sách tất cả các bảng
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        tables = [table[0] for table in cursor.fetchall()]

        # Lọc bỏ các bảng không mong muốn
        tables = [table for table in tables if table not in exclude_tables]

        conn.close()

        # Xuất từng bảng
        success_tables = []
        failed_tables = []

        print(f" Bắt đầu xuất {len(tables)} bảng thành file JSON...")

        for i, table_name in enumerate(tables, 1):
            print(f"  [{i}/{len(tables)}] Đang xuất bảng: {table_name}")

            if export_table_to_json(db_folder,db_path,db_name, table_name, sample_limit):
                success_tables.append(table_name)
                print(f"     Thành công: {table_name}.json")
            else:
                failed_tables.append(table_name)
                print(f"     Thất bại: {table_name}")

        # Tổng kết
        print(f"\nKết quả:")
        print(f"  Thành công: {len(success_tables)} bảng")
        print(f"  Thất bại: {len(failed_tables)} bảng")

        if success_tables:
            print(f"\nFile JSON đã tạo :")
            for table in success_tables:
                print(f"  - {table}.json")

        if failed_tables:
            print(f"\nCác bảng thất bại:")
            for table in failed_tables:
                print(f"  - {table}")

        return {
            "success": success_tables,
            "failed": failed_tables,
            "total": len(tables),
        }

    except Exception as e:
        print(f"Lỗi khi kết nối database: {e}")
        return {"success": [], "failed": [], "total": 0}


def preview_table_structure(db_path):
    """
    Xem trước cấu trúc các bảng trong database
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        tables = [table[0] for table in cursor.fetchall()]

        print(f" Database có {len(tables)} bảng:")

        for table_name in tables:
            column_names, column_types = get_column_info(cursor, table_name)
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row_count = cursor.fetchone()[0]

            print(f"\n   {table_name}:")
            print(f"     - Số dòng: {row_count}")
            print(f"     - Số cột: {len(column_names)}")
            print(f"     - Cột: {', '.join(column_names)}")

        conn.close()

    except Exception as e:
        print(f" Lỗi khi xem trước: {e}")


# Chạy chương trình
if __name__ == "__main__":
    # # Tạo database
    # db_path = create_database()
    # database_path = "perpetrator.sqlite"  # Thay bằng đường dẫn thực tế
    database_path = "baseball_1.sqlite"  # Thay bằng đường dẫn thực tế
    # # # Kiểm tra database
    # # verify_database(database_path)

    # Kiểm tra file database có tồn tại không
    if not os.path.exists(database_path):
        print(f"File database không tồn tại: {database_path}")

    else:
        # Xuất DDL cơ bản
        extract_ddl_to_csv(database_path, "DDL.csv")
        print(" Xem trước cấu trúc database:")
        preview_table_structure(database_path)

        print("\n" + "=" * 50)

        # Xuất tất cả các bảng thành file JSON
        result = export_all_tables_to_json(
            database_path,
            sample_limit=3,  # Lấy 5 dòng mẫu cho mỗi bảng
        )
