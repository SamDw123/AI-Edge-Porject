# We gebruiken de base image die CUDA 12.1 en PyTorch 2.2.0 al heeft
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Installeer systeem dependencies die soms nodig zijn voor CV2/PIL (optioneel, maar veilig)
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installeer de overige python packages
RUN pip install --upgrade pip

# ⭐ FIX 1: Verwijder expliciet de oude Gradio-versie eerst (kan geen kwaad)
RUN pip uninstall -y gradio

# ⭐ FIX 2: Installeer ALLE packages, inclusief de NIEUWE Gradio-versie.
RUN pip install --no-cache-dir -r requirements.txt

COPY Gradio_test.py .

EXPOSE 7860

# Gebruik -u (unbuffered) zodat je print statements direct in je logs ziet
CMD ["python", "-u", "Gradio_test.py"]