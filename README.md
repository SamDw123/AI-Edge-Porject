# SDXL Modular Multi-Control Studio 

Een geavanceerde, modulaire generatieve AI-studio aangedreven door **Stable Diffusion XL (SDXL)**. Deze applicatie stelt gebruikers in staat om afbeeldingen te genereren door gebruik te maken van meerdere gelijktijdige **ControlNet**-lagen en een automatische **Refiner**-stap voor hoogwaardige details.

##  Kernfuncties

* **Multi-Layer ControlNet**: Gebruik twee onafhankelijke ControlNet-lagen tegelijkertijd (bijv. Pose + Depth) om de compositie van je afbeeldingen exact te sturen.
* **SDXL Base + Refiner Workflow**: Automatische verfijning van gegenereerde beelden voor scherpere texturen en realistischere details.
* **Ondersteunde Modi**: 
    * `Pose` (OpenPose)
    * `Canny` (Edge Detection)
    * `Depth` (Midas)
    * `Normal` (NormalBae)
    * `Structure` (Soft-Edge HED)
* **Interactieve UI**: Gebouwd met Gradio voor een intuïtieve browserervaring.
* **Dockerized**: Volledig gecontaineriseerd voor eenvoudige installatie en consistente uitvoering.

## Technologie Stack

* **AI Engine**: Hugging Face Diffusers
* **Modellen**: 
    * Base: `stabilityai/stable-diffusion-xl-base-1.0`
    * Refiner: `stabilityai/stable-diffusion-xl-refiner-1.0`
    * ControlNet: `xinsir/controlnet-union-sdxl-1.0`
* **Vision Tools**: `controlnet_aux` (lllyasviel Annotators)
* **Frontend**: Gradio v4.19.2
* **Runtime**: Python 3.10 + CUDA 12.1

---

## Aan de slag (Installatie & Testen)

Volg deze stappen om het project op je eigen machine of server met een NVIDIA GPU te draaien.

### 1. Vereisten
* NVIDIA GPU met minimaal **16GB - 24GB VRAM**.
* [Docker](https://docs.docker.com/get-docker/) en [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) geïnstalleerd.

### 2. De Docker Image Bouwen
Zorg dat je in de projectmap bent waar de `Dockerfile` en `requirements.txt` staan. Bouw de image met:

``` 
docker build -t sd-gradio . 
```

###  3. De Container Starten

Voer de container uit. We koppelen de Hugging Face cachemap van je host aan de container om te voorkomen dat modellen bij elke herstart opnieuw gedownload moeten worden.

```
docker run -it \
    --gpus all \
    -p 7860:7860 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    sd-gradio
```

### 4. Browser Openen
Zodra de server draait, is de interface bereikbaar via: ```http://localhost:7860``` of via de public URL die gradio aanmaakt. 