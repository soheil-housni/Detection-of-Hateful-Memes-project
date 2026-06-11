FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.8.0 \
        torchvision==0.23.0 \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt &&\
    pip install python-multipart && \
    pip install python-multipart

COPY common_files/ ./common_files/
COPY RESNET_BERT_model/ ./RESNET_BERT_model/

EXPOSE 8000
CMD ["uvicorn", "RESNET_BERT_model.modules.API.app:app", "--host", "0.0.0.0", "--port", "8000"]