import streamlit as st
import pandas as pd
import time
import os 
from sqlite import extract_ddl_to_csv,export_all_tables_to_json
import glob
from setup_vector_chromadb import VietnameseRAGSystem
from retrive_external_context import retrieve_from_collections,save_prompt_context
from reconstruct_data import compress_ddl
from schema_linking import ask_model_sl

# ---- Cấu hình trang ----
st.set_page_config(
    page_title="SQL Agent Chat",
    layout="wide"
)


if "current_db" not in st.session_state:
    st.session_state.current_db = None
db_des= True
# ---- Thanh bên (Sidebar) ----
with st.sidebar:
    
    st.header("Control")

    db_name = st.selectbox(
        "DB Name",
        ("perpetrator", "baseball_1")
    )
    
    db_des= True
    db_folder = os.path.join("db", db_name)
    db_path = os.path.join(db_folder,db_name + ".sqlite")

    if  not os.path.exists(db_path):
        st.error(f" Không tìm thấy file database: {db_path}")
        if st.session_state.current_db:
            current_db = st.session_state.current_db
    else:
        st.session_state.current_db = db_name
    
        schema_path = os.path.join(db_folder,"schema")
        os.makedirs(schema_path, exist_ok=True)

        ddl_path = os.path.join(schema_path,'DDL.csv')
        if  not os.path.exists(ddl_path):
            print(f"File ddl không tồn tại: {ddl_path}")
            extract_ddl_to_csv(db_name, 'DDL.csv')

        json_files = glob.glob(os.path.join(schema_path, "*.json"))

        if json_files:
            pass
        else:
            print("Không có file .json trong thư mục.")
            # Xuất tất cả các bảng thành file JSON
            result = export_all_tables_to_json(
                db_name, 
                sample_limit=3  # Lấy 3 dòng mẫu cho mỗi bảng
            )

    vector_db_path = os.path.join(db_folder, "db_chroma")

    if  not os.path.exists(vector_db_path):
        input_file = os.path.join(db_folder,"db_des", "db_des.txt")
        if  os.path.exists(input_file): 
            st.error(f" Đang tạo vector db db description...")
            rag_system = VietnameseRAGSystem()
            rag_system.setup_database(db_name)
        else:
            st.error(f"DB này không có db description")
            db_des= False

# ---- Giao diện chính ----
st.title("SQL Agent Chat")

# Khởi tạo lịch sử chat trong session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn đã có trong lịch sử
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Kiểm tra nếu nội dung là một dictionary (dành cho trợ lý)
        if isinstance(message["content"], dict):
            st.write(message["content"]["log_header"])
            st.text_area(
                "Log", 
                value=message["content"]["log_body"], 
                height=150, 
                key=f"log_{message['content']['id']}" # Key duy nhất
            )
            st.write(message["content"]["sql_header"])
            st.code(message["content"]["sql_body"], language="sql")
            st.markdown("---")
            st.header(message["content"]["results_header"])
            st.dataframe(message["content"]["results_data"], use_container_width=True, hide_index=True)
        else:
            # Nội dung là text bình thường (dành cho người dùng)
            st.markdown(message["content"])

# ---- Ô nhập liệu chat ----
# Hiển thị ô nhập liệu ở cuối trang
prompt = st.chat_input("Nhập câu hỏi của bạn ở đây...")


if prompt:
    id = int(time.time()) 
    
    results = retrieve_from_collections(db_des,prompt, db_name)
    save_prompt_context(db_des=db_des,results= results,id = id,db_name = db_name)

    compress_ddl(db_name=db_name,id=id,add_description=True, add_sample_rows=True, rm_digits=True, schema_linked=False, clear_long_eg_des=True)
    
    ask_model_sl(task=prompt,id=id,db_name=db_name)

    # 1. Thêm và hiển thị tin nhắn của người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Tạo và hiển thị phản hồi của trợ lý
    with st.chat_message("assistant"):
        # Nội dung phản hồi của trợ lý
        assistant_response_content = {
            "id": int(time.time()), # Tạo ID duy nhất dựa trên thời gian
            "log_header": "self-refine log:",
            "log_body": "Try to run SQL in self-refine...\n\nITER 1:\nITER 2:\n\nConsistent result achieved!",
            "sql_header": "Final SQL:",
            "sql_body": "SELECT * \nFROM PLAPLA \nWHERE 3==D",
            "results_header": "Query results:",
            "results_data": pd.DataFrame({
                'Order#': ['4729237', '2308478', '4729237', '2308478'],
                'City': ['San Francisco', 'New York', 'San Francisco', 'New York'],
                'State': ['CA', 'NY', 'CA', 'NY'],
                'Tracking ID': ['TR-872350', 'TR-102034', 'TR-872350', 'TR-102034']
            })
        }
        
        # Thêm phản hồi vào lịch sử chat
        st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})

        # Hiển thị các thành phần của phản hồi
        st.write(assistant_response_content["log_header"])
        st.text_area(
            "Log", 
            value=assistant_response_content["log_body"], 
            height=150,
            key=f"log_{assistant_response_content['id']}" # Key duy nhất
        )
        st.write(assistant_response_content["sql_header"])
        st.code(assistant_response_content["sql_body"], language="sql")
        st.markdown("---")
        st.header(assistant_response_content["results_header"])
        st.dataframe(assistant_response_content["results_data"], use_container_width=True, hide_index=True)