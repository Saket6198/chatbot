from io import StringIO
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect
import os
from langchain.agents import create_csv_agents
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route("/", methods=["POST", "GET"])
def main():
    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        temperature = 0.2,
        max_tokens=None,
        timeout = None,
        max_retries=3,
    )
    if request.method == "POST":
        csv_data = upload_csv()
        if csv_data:
            agent = create_csv_agents(llm, csv_data, verbose=True)
            return redirect("/")
        else:
            return redirect("/")
    return render_template("upload.html")


def upload_csv():
    file = request.files.get("file")
    if file and file.filename.endswith('csv'):
        file.save(os.path.join("uploads", file.filename))

        file.stream.seek(0)
        csv_data = StringIO(file.stream.read().decode("utf-8"))
        return csv_data

    return None


if __name__ == "__main__":
    app.run(debug=True)