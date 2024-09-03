from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import csv
import codecs

app = FastAPI()

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=3,
)

@app.post("/ask_csv/")
async def create_upload_csv(file: UploadFile = File(...), question: str = Form(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type! Only CSV files allowed.")
    csv_data  = file.read()
    csv_stream = BytesIO(csv_data)

    agent = create_csv_agent(llm, csv_stream, verbose=True)
    result = agent.run(question)

    return {"response": result}



def clean_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def scale_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)