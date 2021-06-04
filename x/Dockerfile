FROM python:3.8

ADD Fine_Tuner_without_KGE.py .
ADD COVID-QA.json .
ADD requirements.txt

RUN pip install -r requirements.txt

CMD ["python", "./Fine_Tuner_without_KGE.py"]