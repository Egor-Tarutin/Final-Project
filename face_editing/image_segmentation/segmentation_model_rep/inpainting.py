from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")

img = Image.open('face.png')
msk = Image.open('mask.png')
img = img.resize(msk.size)
# msk = msk.resize(img.size)

print(img.size)
print(msk.size)

prompt = "zebra's lines"

image = pipe(prompt=prompt, image=img, mask_image=msk).images[0]
image.save("inpaint.png")
