FROM python:3.11-slim

WORKDIR /app

# Copy only subfolder
COPY Titanic-Survival-Prediction-using-ML/ .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
