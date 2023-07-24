import os
import openai
import streamlit as st
from doc_utils import SaveFileToDisk, VectorDB, ProcessDocument, PromptLoader, DisplayDocument
st.set_page_config(layout="wide")

st.title("ProtoGPT")

openai.api_key = os.getenv("OPENAI_API_KEY")

uploaded_file = st.sidebar.file_uploader(
    "Upload a protocol", 
    type=["pdf"]
    )

vector_db = VectorDB()

if uploaded_file:
    file_path = SaveFileToDisk(uploaded_file, vector_db).file_path
    file_name = os.path.basename(file_path)
    if not vector_db.document_exists(file_name):
        documents = ProcessDocument(file_path).load_and_chunk()
        vector_db.add_documents(documents)
        vector_db.add_processed_document(file_name) 
    else:
        st.success(f"The document {file_name} has been added to DB.")

select_file = st.sidebar.selectbox(
    "Select a protocol",
    vector_db.get_document_names(),
    index=0
)

select_variable = st.sidebar.selectbox(
    "Choose variable",
    [
        "num_visits",
        "percent_placebo",
        "biopsy",
        "colonoscopy",
        "mode_of_admin",
        "blood_draw",
        "caregiver",
        "discontinue_treatment",
        "inpatient_trial",
        "trial_type",
        "comparator_drug_mode_admin",
        "comparator_drug_arm"
    ])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = 'gpt-3.5-turbo-0613'

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def compose_prompt(context, content):
    return "Context: " + context + "\n\n" + "Question: " + content + "\n\n"

if select_file:
    prompt = PromptLoader(f"{select_variable}.json").prompt
    context = vector_db.get_context(prompt["messages"][1]["content"], select_file)

    col1, col2, col3 = st.columns([0.6,0.2,0.2])
    with col1:
        DisplayDocument(f"protocols/{select_file}").displayPDF()
    with col2:
        st.subheader("Question")
        st.write(prompt["messages"][1]["content"])    
    with col3:
        st.subheader("Answer")
        prompt["messages"][1]["content"] = compose_prompt(context, prompt["messages"][1]["content"])
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=prompt["messages"],
            stream=True,
            temperature=0.5,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
