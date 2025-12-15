FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip uninstall -y gradio

RUN pip install --no-cache-dir -r requirements.txt

COPY Gradio_test.py .

EXPOSE 7860

CMD ["python", "-u", "Gradio_test.py"]