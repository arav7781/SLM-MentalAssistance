FROM python:3.11

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir llama-cpp-python==0.2.77 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port","8000"]