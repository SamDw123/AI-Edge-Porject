import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import gradio as gr
import os

zôivj^v
# ------------------------------------------------------
# Output folder
# ------------------------------------------------------
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------------------------------
# Load models
# ------------------------------------------------------
print("🔄 Loading SDXL Base...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

print("🔄 Loading SDXL Img2Img Refiner...")
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def generate_image(prompt, seed=42, steps=30, guidance=7.5):
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=1024,
        width=1024
    ).images[0]
    return image

def edit_generated_image(base_image, edit_prompt, seed=42, steps=30, guidance=7.5, strength=0.5):
    generator = torch.Generator("cuda").manual_seed(seed)
    base_image = base_image.resize((1024, 1024))
    edited_image = refiner(
        prompt=edit_prompt,
        image=base_image,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength
    ).images[0]
    return edited_image

# ------------------------------------------------------
# Gradio UI
# ------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## SDXL Photo Generator & Editor")

    with gr.Tab("Generate & Edit Stepwise"):
        prompt_input = gr.Textbox(label="Initial Prompt")
        seed_input = gr.Number(value=42, label="Seed")
        steps_input = gr.Slider(10, 100, value=30, step=1, label="Steps")
        guidance_input = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
        generate_btn = gr.Button("Generate Image")
        generated_output = gr.Image(label="Generated Image")
        
        # State component om de gegenereerde afbeelding op te slaan
        generated_state = gr.State()
        
        generate_btn.click(
            fn=lambda prompt, seed, steps, guidance: (generate_image(prompt, seed, steps, guidance), generate_image(prompt, seed, steps, guidance)),
            inputs=[prompt_input, seed_input, steps_input, guidance_input],
            outputs=[generated_output, generated_state]
        )
        
        gr.Markdown("### Edit the Generated Image")
        edit_prompt_input = gr.Textbox(label="Edit Prompt")
        strength_input = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Edit Strength")
        edit_btn = gr.Button("Edit Generated Image")
        edited_output = gr.Image(label="Edited Image")
        
        edit_btn.click(
            fn=edit_generated_image,
            inputs=[generated_state, edit_prompt_input, seed_input, steps_input, guidance_input, strength_input],
            outputs=edited_output
        )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
