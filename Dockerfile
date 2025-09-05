FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app
COPY requirements.txt ./
RUN pip --no-cache-dir install -r requirements.txt

COPY app.py ./

ENV EPOCHS=1
CMD ["python", "app.py"]
