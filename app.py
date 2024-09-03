from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st


def main():
    global agent
    load_dotenv()

    # Load the API_keys
    # if os.getenv("GOOGLE_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    #     print("GOOGLE_API_KEY is not set")
    #     exit(1)
    # else:
    #     print("GOOGLE_API_KEY is set")


    st.set_page_config(
        page_title="Ask your CSV",
        layout="wide",
    )

    st.header("Ask your CSV")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        user_question = st.text_input("Ask a question about your CSV: ")
        llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature = 0.2,
            max_tokens=None,
            timeout = None,
            max_retries = 3,
        )
        agent = create_csv_agent(
            llm,
            csv_file,
            verbose=True,
            allow_dangerous_code=True,
        )
        if user_question is not None and user_question != "":
            # st.write(f'Your Question: {user_question}')
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))

if __name__ == "__main__":
    main()