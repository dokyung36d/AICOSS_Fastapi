FROM python:latest

WORKDIR /code

COPY . /code
 
RUN pip install -r /code/requirements.txt


CMD uvicorn --host=0.0.0.0 --port 8000 app.main:app

 
COPY ./app /code/app
 
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
