import streamlit as st
from streamlit_chat import message
from src.utils.load_config import LoadConfig
from src.utils.app_utils import Apputils

cfg = LoadConfig()

if "user_queries" not in st.session_state:
    st.session_state["user_queries"] = []
if "llm_responses" not in st.session_state:
    st.session_state["llm_responses"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

response_container = st.container()  # for chat history
container = st.container()  # for text box
container.markdown("""
    <style>
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            background-color: #f5f5f5;
            border-top: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_query = st.text_area("You:", key='input')
        submit_button = st.form_submit_button(label='Submit')

    if user_query and submit_button:
        chat_history = f"#Chat History :\n{st.session_state['chat_history'][-2:]}\n\n"
        query = f"User new question :\n {user_query} \n\n"
        messages = [
            {
                "role": "system",
                "content": str(cfg.llm_function_caller_system_role)
            },
            {
                "role": "user",
                "content": chat_history + query
            }
        ]

        first_llm_response = Apputils.ask_llm_function_caller(
            gpt_model=cfg.gpt_model,
            temperature=cfg.temperature,
            messages=messages,
            function_json_list=Apputils.wrap_functions()
        )

        st.session_state["user_queries"].append(user_query)

        if hasattr(first_llm_response.choices[0].message,"tool_calls"):
            print("Called function:",first_llm_response.choices[0].message.tool_calls[0].function.name)
            web_search_result = Apputils.execute_json_function(first_llm_response)
            web_search_results = f"\n Web Search Results : \n {str(web_search_result)} \n\n"
            messages = [
                {
                    "role":"system",
                    "content": str(cfg.llm_function_caller_system_role)
                },
                {
                    "role":"user",
                    "content": chat_history + web_search_results + query
                }
            ]
            second_llm_response = Apputils.ask_llm_chatbot(
                gpt_model=cfg.gpt_model,
                temperature=cfg.temperature,
                messages=messages
            )
            st.session_state["llm_responses"].append(second_llm_response.choices[0].message.content)
            chat_history = (f"## User query: {query}", f"## Response: {second_llm_response.choices[0].message.content}")
            st.session_state['chat_history'].append(chat_history)
        else:
            chat_history = (f"## User query: {query}", f"## Response: {first_llm_response.choices[0].message.content}")
            st.session_state['chat_history'].append(chat_history)
            st.session_state["llm_responses"].append(first_llm_response.choices[0].message.content)


if st.session_state['llm_responses']:
    with response_container:
        for i in range(len(st.session_state['llm_responses'])):
            message(st.session_state["user_queries"][i],
                    is_user=True,
                    key=str(i) + '_user',
                    )
            message(st.session_state["llm_responses"][i],
                    key=str(i),
                    )

