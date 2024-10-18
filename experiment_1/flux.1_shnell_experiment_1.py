#!/home/raze/flux/.venv/bin/python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

aura = [
    # 'mysterious',
      'ancient', 'beautiful', 'cozy', 'dry', 
      'inspiring', 'lively', 'majestic', 'organized', 'soothing']
aura.sort()

for i in range(7,8):
    for a in aura[4:]:
        prompt = f"A {a} underwater library"
        # prompt = "A boy reading in a mysterious underwater library"
        print(prompt)
        images = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(i),
            # num_images_per_prompt=100,
            # aspect_ratio="16:9",
            height=1080,
            width=1920,
            ).images[0]
    # for i in range(1,len(images.images)+1):
        # images.images[i].save(f"mysterious_library-{i}.png")
        images.save(f"{a}_library-{i}.png")

for i in range(8,13):
    for a in aura:
        prompt = f"A {a} underwater library"
        # prompt = "A boy reading in a mysterious underwater library"
        print(prompt)
        images = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(i),
            # num_images_per_prompt=100,
            # aspect_ratio="16:9",
            height=1080,
            width=1920,
            ).images[0]
    # for i in range(1,len(images.images)+1):
        # images.images[i].save(f"mysterious_library-{i}.png")
        images.save(f"{a}_library-{i}.png")
