import streamlit as st
import pandas as pd
import time
import os
import sys
import glob
import io
import gzip
import base64
import chromadb
from datetime import datetime
from dotenv import load_dotenv
# Thêm thư mục gốc vào path để import các module
load_dotenv()
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if root not in sys.path:
    sys.path.insert(0, root)

# Import các module từ backend
try:
    from database_manager import DatabaseManager
    from llm_client import OpenAIClient, create_ollama_client,GPTChat,GPTChat_sl
    from agent import Agent
    from pre.sqlite import extract_ddl_to_csv, export_all_tables_to_json
    from pre.setup_vector_chromadb import VietnameseRAGSystem
    from pre.retrive_external_context import retrieve_from_collections, save_prompt_context,check_if_question_relevant,VietnameseEmbedding
    from pre.reconstruct_data import compress_ddl
    from pre.schema_linking import ask_model_sl
    from pre.build_index import main
    # from pre.query_lsh import LSHChromaNormalizer
except ImportError as e:
    st.error(f"Lỗi import module: {e}")
    st.stop()
# ---- Cấu hình trang ----
st.set_page_config(
    page_title="SQL Agent Chat",
    layout="wide"
)

# Khởi tạo session state
if "current_db" not in st.session_state:
    st.session_state.current_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "dbman" not in st.session_state:
    st.session_state.dbman = None
if  "embedding_model" not in st.session_state:
    st.session_state.embedding_model = VietnameseEmbedding()

# ---- Khởi tạo LLM Client ----
@st.cache_resource
def init_llm_client():
    """Khởi tạo LLM client"""
    system_prompt = """You are a data science expert that can write excellent SQL queries. Below, you are provided with a database schema, a natural question, and some necessary context. Your task is to understand the schema, the context, and generate a valid SQL query to answer the question.

    Instructions:
    - Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
    - The generated query should return all of the information asked in the question without any missing or extra information.
    - Before generating the final SQL query, please think through the steps of how to write the query.
    - For string-matching scenarios, if the string is decided, don't use fuzzy query. e.g. Get the object's title contains the word "book". However, if the string is not decided, you may use fuzzy query and ignore upper or lower case. e.g. Get articles that mention "education".
    - If the task description does not specify the number of decimal places, retain all decimals to four places.
    - For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he's to he%s)
    - When asked something without stating name or id, return both of them. e.g. Which products ...? The answer should include product_name and product_id.
    - When asked percentage decrease, you should return a positive value. e.g. How many percentage points in 2021 decrease compared to ...? The answer should be a positive value indicating the decreased number. Try to use ABS().
    - If asked two tables, you should reply with the last one instead of combining two tables. e.g. Identifying the top five states ... examine the state that ranks fourth overall and identify its top five counties. You should only answer top five counties.

    Take a deep breath and think step by step to find the correct SQL query.
    """
    
    try:
        # Sử dụng OpenAI client (có thể thay đổi thành Ollama nếu cần)
        # client = OpenAIClient(
        #     model="gpt-4.1",  # Thay đổi model phù hợp
        #     temperature=1,
        #     max_context_length=200_000,
        #     system_prompt=system_prompt,
        # )
        client =  GPTChat(system_prompt=system_prompt,temperature=1)
        client_sl =  GPTChat_sl(system_prompt="",temperature=0)
        return client,client_sl
    except Exception as e:
        st.error(f"Lỗi khởi tạo LLM client: {e}")
        return None

# ---- Hàm setup database ----
def setup_database(db_name):
    """Setup database và các file cần thiết"""
    db_des = True
    db_folder = os.path.join("pre/db", db_name)
    db_path = os.path.join(db_folder, db_name + ".sqlite")
    
    
    if not os.path.exists(db_path):
        st.error(f"Không tìm thấy file database: {db_path}")
        return None, None, False
    
    if not os.path.exists(os.path.join("./lsh_semantic", f"{db_name}_lsh_buckets.sqlite") ):
        st.info(f"Đang build index cho dữ liệu database: {db_name}")
        main(db_path=db_path,db_name=db_name,model=st.session_state.embedding_model)
        st.success(" Build xong index!")

    # Tạo schema path
    schema_path = os.path.join(db_folder, "schema")
    os.makedirs(schema_path, exist_ok=True)
    
    # Kiểm tra và tạo DDL.csv
    ddl_path = os.path.join(schema_path, 'DDL.csv')
    if not os.path.exists(ddl_path):
        st.info("Đang tạo DDL.csv...")
        try:
            extract_ddl_to_csv(db_folder,db_path,db_name, 'DDL.csv')
            st.success("Đã tạo DDL.csv thành công!")
        except Exception as e:
            st.error(f"Lỗi tạo DDL.csv: {e}")
    
    # Kiểm tra và tạo JSON files
    json_files = glob.glob(os.path.join(schema_path, "*.json"))
    if not json_files:
        st.info("Đang xuất bảng thành JSON...")
        try:
            result = export_all_tables_to_json(db_folder,db_path,
                db_name, 
                sample_limit=3
            )
            st.success("Đã xuất JSON thành công!")
        except Exception as e: 
            st.error(f"Lỗi xuất JSON: {e}")

    rag_system = VietnameseRAGSystem(st.session_state.embedding_model)
    chroma_client = chromadb.PersistentClient(path=os.path.join(db_folder, "db_chroma"))
    
    if not os.path.exists(os.path.join(db_folder,"prompts", db_name + ".txt")):
        st.info("Đang Compress schema...")
        try:
            # Compress DDL
            compress_ddl(db_folder,db_path,
            db_name=db_name,
            id=id,
            add_description=True,
            add_sample_rows=True,
            rm_digits=True,
            schema_linked=False,
            clear_long_eg_des=True,log_callback=None
        )   
             
            st.success("Đã Compress schema thành công!")
        except Exception as e:
            st.error(f"Lỗi Compress schema: {e}")
        st.info("Đang set up vector db cho schema db ...")
        try:
            rag_system.setup_vector_db_schema_db(db_name,db_folder)
   
            st.success("Đã set up vector db thành công!")
        except Exception as e:
            st.error(f"Lỗi set up vector db: {e}")
        
        
    # Kiểm tra vector database
    
    if "db_des" not in [col.name for col in chroma_client.list_collections()]:
        input_file = os.path.join(db_folder, "db_des", "db_des.txt")
        if os.path.exists(input_file):
            st.info("Đang tạo vector database...")
            try:
                
                rag_system.setup_database(db_name,db_folder)
                st.success("Đã tạo vector database thành công!")
            except Exception as e:
                st.error(f"Lỗi tạo vector database: {e}")
        else:
            # st.warning("Database này không có mô tả")
            db_des = False
    
    return db_folder, db_path, db_des

# ---- Hàm khởi tạo agent ----
def init_agent(db_path, db_name, client):
    """Khởi tạo database manager và agent"""
    try:
        # Khởi tạo database manager
        dbman = DatabaseManager()
        dbman.start_sqlite(db_path)

        # Khởi tạo agent
        agent = Agent(
            db_path=os.path.dirname(db_path),
            db_id=os.path.basename(db_path),
            db_manager=dbman,
            client=client,
        )
        
        # Health check
        agent.health_check()
        
        return agent, dbman
    except Exception as e:
        st.error(f"Lỗi khởi tạo agent: {e}")
        return None, None

# ---- Hàm xử lý câu hỏi ----
def process_question(id,db_folder,db_path,question, db_name, db_des,client,log_callback=None):
    """Xử lý câu hỏi và trả về kết quả"""
    try:
        
        # Retrieve context
        desc_exemplars,_ = retrieve_from_collections(st.session_state.embedding_model,db_folder,db_path,db_des, question, db_name,log_callback=log_callback)
        
        # Save prompt context
        save_prompt_context(
            db_des,
            db_folder,
            db_path,
            results=desc_exemplars, 
            id=id, 
            db_name=db_name,log_callback=log_callback
        )
        
        # # Compress DDL
        # compress_ddl(db_folder,db_path,
        #     db_name=db_name,
        #     id=id,
        #     add_description=True,
        #     add_sample_rows=True,
        #     rm_digits=True,
        #     schema_linked=False,
        #     clear_long_eg_des=True,log_callback=log_callback
        # )
        # Schema linking
        ask_model_sl(
            db_folder,db_path,
            task=question,
            id=id,
            db_name=db_name,
            chat_session=client,
            log_callback=log_callback
        )
        
        # # Đọc final context prompt
        # # db_folder = os.path.join("pre/db", db_name)
        # final_context_prompt_path = os.path.join(
        #     db_folder, 'final_context_prompts', str(id) + '.txt'
        # )
        
        # prompt_template = ""
        # with open(final_context_prompt_path, 'r', encoding='utf-8') as f:
        #     prompt_template = f.read()
        
        # # Chạy self-refine
        # self_refine_result = agent.self_refine(
        #     question=question,
        #     base_prompt=prompt_template
        # )
        
        return   id  
    except Exception as e:
        st.error(f"Lỗi xử lý câu hỏi: {e}")
        return None, None
def self_refine(id,db_folder,question,agent,log_callback=None):
    # Đọc final context prompt
    # db_folder = os.path.join("pre/db", db_name)
    print(id)
    final_context_prompt_path = os.path.join(
        db_folder, 'final_context_prompts', str(id) + '.txt'
    )
    
    prompt_template = ""
    with open(final_context_prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Chạy self-refine
    self_refine_result = agent.self_refine(
        question=question,
        base_prompt=prompt_template,
        log_callback=log_callback
    )
    return self_refine_result

# def update_log(msg, logs,id,placeholder):
#     logs.append(msg)
#     placeholder.text_area(
#         "Processing Log", 
#         value="\n".join(logs), 
#         height=200, 
#         disabled=True
#     )   

def update_log(msg, logs, id, placeholder):
    if "logs_dict" not in st.session_state:
        st.session_state.logs_dict = {}
    if id not in st.session_state.logs_dict:
        st.session_state.logs_dict[id] = []
        
    st.session_state.logs_dict[id].append(msg)
    logs.append(msg)

    placeholder.text_area(
        "Processing Log", 
        value="\n".join(logs), 
        height=200, 
        disabled=True,
        # key=f"log_placeholder_{id}"
    )


def decode_dataframe(result):
    """Giải nén dict kết quả thành DataFrame nếu cần"""
    if not result:
        return None
    if isinstance(result, dict) and "df_csv_data" in result:
        compressed_bytes = base64.b64decode(result["df_csv_data"])
        csv_data = gzip.decompress(compressed_bytes).decode("utf-8")
        return pd.read_csv(io.StringIO(csv_data))
    return result

def get_available_databases():
    db_root = "pre/db"
    return [
        name for name in os.listdir(db_root)
        if os.path.isdir(os.path.join(db_root, name))
    ]

client,client_sl = init_llm_client()
         
# ---- Thanh bên (Sidebar) ----
with st.sidebar:
    st.header(" Control Panel")
    # Upload database mới
    uploaded_db = st.file_uploader("📤 Tải lên SQLite DB mới", type=["sqlite", "db"],key="file1")
    if uploaded_db is not None:
        new_db_name = st.text_input("Tên thư mục lưu DB:", value=uploaded_db.name.split(".")[0])
        if st.button("➕ Thêm DB"):
            new_db_folder = os.path.join("pre/db", new_db_name)
            os.makedirs(new_db_folder, exist_ok=True)
            db_path = os.path.join(new_db_folder, f"{new_db_name}.sqlite")
            
            with open(db_path, "wb") as f:
                f.write(uploaded_db.read())
            
            st.success(f"Đã thêm DB mới: {new_db_name}")
            st.rerun()  # Load lại trang để hiện DB mới

    available_dbs = get_available_databases()
    # Chọn database
    db_name = st.selectbox(
        " Chọn Database",
        available_dbs,
        help="Chọn database để làm việc"
    )
    # Setup database khi thay đổi
    if st.session_state.current_db != db_name:
        st.session_state.current_db = db_name
        st.session_state.agent = None
        st.session_state.dbman = None
        
        with st.spinner("Đang setup database..."):
            db_folder, db_path, db_des = setup_database(db_name)
            
            if db_path:
                st.session_state.db_folder = db_folder
                st.session_state.db_path = db_path
                st.session_state.db_des = db_des
                st.success(f" Database {db_name} đã sẵn sàng!")

    
    if not st.session_state.db_des:
        st.warning("Database này chưa có mô tả.")
        # breakpoint()
        uploaded_desc = st.file_uploader("📄 Tải lên file mô tả (.txt)", type=["txt"],key="file2")
        # print(uploaded_desc)
        if uploaded_desc is not None:
            print("1")
            db_des_folder = os.path.join(st.session_state.db_folder, "db_des")
            os.makedirs(db_des_folder, exist_ok=True)
            desc_path = os.path.join(db_des_folder, "db_des.txt")
            
            with open(desc_path, "wb") as f:
                f.write(uploaded_desc.read())
            
            st.success("Đã thêm mô tả cho database!")
            
            # Tạo lại vector database sau khi có mô tả
            try:
                st.info("Đang tạo vector database...")
                rag_system = VietnameseRAGSystem(st.session_state.embedding_model)
                rag_system.setup_database(db_name,st.session_state.db_folder)
                st.success("Vector database đã được tạo thành công!")
                st.session_state.db_des = True
            except Exception as e:
                st.error(f"Lỗi tạo vector database: {e}")            
    # Khởi tạo LLM client và agent
    if st.button(" Khởi tạo Agent"):
        with st.spinner("Đang khởi tạo LLM client và agent..."):
            # client = init_llm_client()
            print(st.session_state.db_path)
            if client:
                agent, dbman = init_agent(
                    st.session_state.db_path, 
                    db_name, 
                    client
                )
                if agent:
                    st.session_state.agent = agent
                    st.session_state.dbman = dbman
                    st.success(" Agent đã sẵn sàng!")
                else:
                    st.error(" Lỗi khởi tạo agent")
            else:
                st.error(" Lỗi khởi tạo LLM client")
    
    # Hiển thị trạng thái
    st.markdown("---")
    st.subheader(" Status")
    if st.session_state.current_db:
        st.success(f"DB: {st.session_state.current_db}")
    if st.session_state.agent:
        st.success("Agent:  Ready")
    else:
        st.warning("Agent:  Not initialized")

# ---- Giao diện chính ----
st.title("🤖 SQL Agent Chat")
st.markdown("Hỏi đáp với database bằng ngôn ngữ tự nhiên")

# Kiểm tra agent đã được khởi tạo chưa
if not st.session_state.agent:
    st.warning(" Vui lòng khởi tạo Agent từ sidebar trước khi sử dụng!")
    st.stop()

# Hiển thị các tin nhắn đã có trong lịch sử
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            # Hiển thị phản hồi của assistant
            st.subheader("📋 Log:")
            st.text_area(
                "Processing Log", 
                value=message["content"]["log_body"], 
                height=150, 
                key=f"log_{message['content']['id']}",
                disabled=True
            )
            
            st.subheader("🔍 Final SQL:")
            st.code(message["content"]["sql_body"], language="sql")
            
            st.markdown("---")
            st.subheader(" Query Results:")
            if message["content"]["results_data"] is not None:
                st.dataframe(
                    message["content"]["results_data"], 
                    use_container_width=True, 
                    hide_index=True
                )
            else:
                st.info("Không có dữ liệu để hiển thị")
        else:
            # Tin nhắn của user
            st.markdown(message["content"])




# ---- Ô nhập liệu chat ----
prompt = st.chat_input("💬 Nhập câu hỏi của bạn ở đây...")

if prompt:
    # Thêm tin nhắn của user
    with st.chat_message("user"):
        st.markdown(prompt)

    is_relevant  = check_if_question_relevant(st.session_state.embedding_model,st.session_state.db_folder,st.session_state.db_path,st.session_state.db_des, prompt, db_name,top_k=5,distance_threshold=0.57)
    
    
    if is_relevant:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Xử lý và trả lời
        with st.chat_message("assistant"):
            with st.spinner("🤔 Scheama linking and Self refine..."):
                st.subheader("📋 Log:")
                # log_placeholder = st.empty()  # Placeholder cho log real-time
                expander_placeholder = st.empty()
                logs = []  # Lưu log tạm thời
                id = int(time.time())
                thinking_container = st.container()
                if "expander_states" not in st.session_state:
                     st.session_state.expander_states = {}

                def update_thinking_log(msg, logs, id, container):
                    """Update log trong thinking section"""
                    
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    # Thêm icon tương ứng với loại message
                    if "thành công" in msg.lower() or "✅" in msg:
                        icon = "✅"
                    elif "lỗi" in msg.lower() or "❌" in msg:
                        icon = "❌"
                    elif "đang" in msg.lower():
                        icon = "⏳"
                    else:
                        icon = "💭"
                        
                    formatted_msg = f"\n{icon} [{timestamp}] {msg}"
                    logs.append(formatted_msg)
                    

                    # Cập nhật thinking section
                    with container:
                        with expander_placeholder.expander(f"{msg} ", expanded=True):
                            # st.session_state.expander_states[expander_key] = True
                            # Custom CSS cho log
                            # st.markdown("""
                            # <style>
                            # .thinking-log {
                            #     background-color: #0E1117   ;
                            #     border-left: 4px solid #007acc;
                            #     padding: 10px;
                            #     border-radius: 5px;
                            #     font-family: 'Verdana', monospace;
                            #     font-size: 12px;
                            #     max-height: 3000px;
                            #     overflow-y: auto;
                            # }
                            # </style>
                            # """, unsafe_allow_html=True)
                            
                            # # Hiển thị log với markdown
                            # log_text = "<br>".join(logs)
                            # st.markdown(f'<div class="thinking-log"><pre>{log_text}</pre></div>', 
                            #         unsafe_allow_html=True)

                            log_text = "\n".join(logs)
                            st.markdown(log_text)
                
                _ = process_question(
                    id,
                    st.session_state.db_folder,
                    st.session_state.db_path,
                    prompt, 
                    st.session_state.current_db, 
                    st.session_state.db_des, 
                    client_sl,
                    log_callback=lambda msg: update_thinking_log(msg, logs, id, thinking_container)
                )
                
                success, final_result, final_sql, log_text = self_refine(
                    id,
                    st.session_state.db_folder,
                    prompt,
                    st.session_state.agent,
                    log_callback=lambda msg: update_thinking_log(msg, logs, id, thinking_container)
                )            
                # Xử lý câu hỏi
                # _ = process_question(
                #     id,
                #     st.session_state.db_folder,
                #     st.session_state.db_path,
                #     prompt, 
                #     st.session_state.current_db, 
                #     st.session_state.db_des, 
                #     client_sl,
                #     log_callback=lambda msg: update_log(msg,logs,id, log_placeholder)
                # )
                
                # success, final_result, final_sql, log_text = self_refine(
                #     id,
                #     st.session_state.db_folder,
                #     prompt,
                #     st.session_state.agent,
                #     log_callback=lambda msg: update_log(msg, logs,id, log_placeholder)
                # )


                if success:
                    # Tạo nội dung phản hồi
                    assistant_response = {
                        "id": id,
                        "log_body": "",
                        "sql_body": final_sql or "-- Không có SQL được tạo",
                        "results_data": final_result
                    }
                    
                    # # Hiển thị log cuối cùng
                    
                    # update_log( "Đã xử lý thành công!", logs,id,log_placeholder)
                    update_thinking_log(" Đã xử lý thành công!", logs, id, thinking_container)
                    # st.text_area(
                    #     "Processing Log", 
                    #     value="\n".join(logs), 
                    #     height=150, 
                    #     key=f"current_log_{assistant_response['id']}",
                    #     disabled=True
                    # )
                    assistant_response["log_body"] = "\n".join(logs)

                    st.subheader(" Final SQL:")
                    st.code(assistant_response["sql_body"], language="sql")

            

                    # --- Trong phần hiển thị ---
                    st.markdown("---")
                    st.subheader(" Query Results:")
                    results_data = decode_dataframe(assistant_response["results_data"])

                    if results_data is not None:
                        if isinstance(results_data, pd.DataFrame):
                            st.dataframe(
                                results_data, 
                                use_container_width=True, 
                                hide_index=True
                            )
                        else:
                            st.write(results_data)
                    else:
                        st.info("Không có dữ liệu để hiển thị")
                    assistant_response["results_data"] = results_data

                    # Lưu vào lịch sử
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })

                else:
                    st.error("Có lỗi xảy ra khi xử lý câu hỏi!")
    else:
        with st.chat_message("assistant"):
            st.warning("Câu hỏi này không liên quan đến cơ sở dữ liệu hiện tại.")
            st.info("💡 Bạn có thể:")
            st.markdown("""
            - Đặt câu hỏi khác liên quan đến **database hiện tại**.
            - Hoặc chọn lại **database** khác phù hợp hơn.
            """)
            
            # Gợi ý nếu muốn
            # related_topics = get_related_topics(db_folder, top_k=3)
            related_topics=["Cầu thủ 'P002' đã đánh được bao nhiêu home run trong giai đoạn hậu mùa giải năm 2024?","Cầu thủ 'P003' đã học tại trường đại học nào vào năm 2010?","ERA trung bình của các cầu thủ trong đội 'T001' cho mùa giải 2024 là bao nhiêu?"]
            if related_topics:
                st.subheader("🔍 Chủ đề gợi ý:")
                for topic in related_topics:
                    st.write(f"- {topic}")  
# ---- Footer ----
st.markdown("---")
st.markdown(" **SQL Agent Chat** - Powered by AI")