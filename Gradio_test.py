import torch
import os
from PIL import Image
import gradio as gr
import gc 

from diffusers import (
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
from controlnet_aux import (
    OpenposeDetector, 
    CannyDetector,
    MidasDetector,
    NormalBaeDetector,
    HEDdetector 
)

# Initialisatie & Modellen
output_folder = "images"
os.makedirs(output_folder, exist_ok=True) 

print("Laden van detectoren...")
openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
canny_detector = CannyDetector()
midas_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
normal_detector = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")


print("Laden van SDXL Union ControlNet...")
union_model = ControlNetModel.from_pretrained(
    "xinsir/controlnet-union-sdxl-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)

print("Laden van Pipelines...")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=[union_model, union_model], 
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Laad en configureer de Refiner
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
refiner.vae.enable_tiling()
refiner.vae.enable_slicing()

# Modulaire Logica

def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def process_control(image, mode, strength): 
    # logica voor ControlNet detectors
    if image is None or mode == "None" or strength == 0:
        return Image.new("RGB", (1024, 1024), "black")
    
    if mode == "Pose":
        res = openpose_detector(image)
    elif mode == "Canny":
        res = canny_detector(image, low_threshold=100, high_threshold=200) 
    elif mode == "Depth":
        res = midas_detector(image)
    elif mode == "Normal":
        res = normal_detector(image)
    elif mode == "Structure": 
        res = hed_detector(image, safe=False)
    else:
        res = Image.new("RGB", (1024, 1024), "black")
    
    return res.resize((1024, 1024), Image.LANCZOS)

def run_modular_control(prompt, seed, steps, cfg, 
                        img1, mode1, strength1,
                        img2, mode2, strength2):
    
    generator = torch.Generator("cuda").manual_seed(int(seed))
    
    ctrl1 = process_control(img1, mode1, strength1)
    ctrl2 = process_control(img2, mode2, strength2)
    
    scales = [float(strength1), float(strength2)]
    
    # Genereren met ControlNet (geeft ruwe output)
    image = pipe(
        prompt=prompt,
        image=[ctrl1, ctrl2],
        controlnet_conditioning_scale=scales, 
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        generator=generator,
        height=1024,
        width=1024,
        output_type="pil" 
    ).images[0]
    
    # Verbeteren met Refiner
    # De Refiner gebruikt de output van de Base Pipeline als input
    refined_image = refiner(
        prompt=prompt,
        image=image, 
        num_inference_steps=10, # De Refiner heeft minder stappen nodig
        generator=generator,
        denoising_start=0.8 # Begin met denoising bij 80% van het proces (standaard voor refinement)
    ).images[0]
    
    cleanup()
    return refined_image, ctrl1, ctrl2 # Retourneer de verfijnde afbeelding


# Gradio UI 
CONTROL_MODES = ["None", "Pose", "Canny", "Depth", "Normal", "Structure"]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("#SDXL Modular Multi-Control Studio")

    with gr.Tab("Custom Multi-Control"):

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", placeholder="Bijv: A medieval knight standing in a vast forest")
                seed = gr.Number(value=42, label="Seed")
                steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                cfg = gr.Slider(1, 15, value=7.0, label="Guidance Scale")
                run_btn = gr.Button("🔥 Genereer Afbeelding", variant="primary")

            with gr.Column(scale=1):
                output_img = gr.Image(label="Eindresultaat")

        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("### Laag 1")
                in1 = gr.Image(label="Referentie Image", type="pil")
                type1 = gr.Dropdown(CONTROL_MODES, value="Structure", label="Type")
                str1 = gr.Slider(0.0, 2.0, value=1.0, label="Strength")
                out1 = gr.Image(label="Preview 1", interactive=False)

            with gr.Column(variant="panel"):
                gr.Markdown("### Laag 2")
                in2 = gr.Image(label="Referentie Image", type="pil")
                type2 = gr.Dropdown(CONTROL_MODES, value="Pose", label="Type")
                str2 = gr.Slider(0.0, 2.0, value=1.2, label="Strength")
                out2 = gr.Image(label="Preview 2", interactive=False)

        run_btn.click(
            fn=run_modular_control,
            inputs=[prompt, seed, steps, cfg, in1, type1, str1, in2, type2, str2],
            outputs=[output_img, out1, out2]
        )
        
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)