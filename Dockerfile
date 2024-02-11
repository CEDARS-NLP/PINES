FROM pytorch/pytorch:latest

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .
# RUN poetry install 

EXPOSE 8036

CMD ["uvicorn", "pines:app", "--host", "0.0.0.0", "--port", "8036"]