import torch
import numpy as np
import os
import cv2

from PIL import Image
from torchvision import transforms
from django.core.cache import cache
from .model import BiSeNet
from diffusers import StableDiffusionInpaintPipeline


def get_inpainting_mask(seg_mask, x_coord, y_coord):
    to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])
    seg_mask = to_tensor(seg_mask)
    num_class = seg_mask[0, int(y_coord), int(x_coord)]
    inp_mask = seg_mask.clone().detach()
    inp_mask = torch.where(inp_mask == num_class, torch.tensor(255), torch.tensor(0))
    inp_arr_mask = np.array(inp_mask, dtype=int).squeeze(0).astype('uint8')
    inp_image = Image.fromarray(inp_arr_mask)
    return inp_image


def get_generated_image(image, inp_mask, prompt):
    image = image.resize((512, 512), Image.Resampling.BILINEAR)
    # generator = torch.Generator("cpu").manual_seed(1024)
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    gen_image = pipe(prompt=prompt,
                     image=image,
                     mask_image=inp_mask,
                     num_inference_steps=5  # ,
                     # generator=generator
                     ).images[0]
    return gen_image


def get_segmentation_mask(image):
    # define a device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    model = cache.get('face_segmentation_model')
    if model is None:
        # define paths
        weight_path = 'image_segmentation/segmentation_model_rep/res/cp'
        weight_file_name = 'model.pth'
        # define model params
        n_classes = 19
        model = BiSeNet(n_classes=n_classes)
        model_path = os.path.normpath(os.path.join(weight_path, weight_file_name))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        cache.set('face_segmentation_model', model)

    model.to(device)
    model.eval()
    # load transform
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load the image and apply any preprocessing steps
    with torch.no_grad():
        image = image.resize((512, 512), Image.Resampling.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = model(img)[0]
        mask = out.squeeze(0).cpu().numpy().argmax(0)
    # form.save()
    mask = mask.astype('uint8')
    mask_image = Image.fromarray(mask)
    return mask_image


def get_colorful_mask(image, mask):
    part_colors = [[255, 0, 0],
                   [255, 85, 0],
                   [255, 170, 0],
                   [255, 0, 85],
                   [255, 0, 170],
                   [0, 255, 0],
                   [85, 255, 0],
                   [170, 255, 0],
                   [0, 255, 85],
                   [0, 255, 170],
                   [0, 0, 255],
                   [85, 0, 255],
                   [170, 0, 255],
                   [0, 85, 255],
                   [0, 170, 255],
                   [255, 255, 0],
                   [255, 255, 85],
                   [255, 255, 170],
                   [255, 0, 255],
                   [255, 85, 255],
                   [255, 170, 255],
                   [0, 255, 255],
                   [85, 255, 255],
                   [170, 255, 255]]

    parsing_anno = np.array(mask)
    image = image.resize(mask.size)
    image = np.array(image)
    vis_im = image.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return Image.fromarray(vis_im)


if __name__ == '__main__':
    pass
