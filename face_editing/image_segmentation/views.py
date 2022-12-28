from django.shortcuts import render, redirect
import datetime

from PIL import Image
from .forms import ImageUploadForm
from .segmentation_model_rep.inp_mask_from_seg import get_inpainting_mask
from .segmentation_model_rep.inp_mask_from_seg import get_segmentation_mask
from .segmentation_model_rep.inp_mask_from_seg import get_generated_image
from .segmentation_model_rep.inp_mask_from_seg import get_colorful_mask


save_image_path = 'statics/save_images/'
save_mask_path = 'statics/save_masks/'
save_gen_image_path = 'statics/save_gen_images/'
save_inp_mask_path = 'statics/save_inp_masks/'
save_color_mask_path = 'statics/save_color_masks/'


def segmentation_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image = Image.open(image)
            mask_image = get_segmentation_mask(image)
            current_date = datetime.datetime.now().date().__str__()
            color_mask = get_colorful_mask(image, mask_image)
            color_mask.save('{}{}.png'.format(save_color_mask_path, current_date))
            mask_image.save('{}{}.png'.format(save_mask_path, current_date))
            image.save('{}{}.png'.format(save_image_path, current_date))

            # Render the template for selecting the inpainting mask
            return render(request, 'image_segmentation/choose_seg_template.html',
                          {'image_path': '{}{}.png'.format(save_image_path, current_date),
                           'mask_path': '{}{}.png'.format(save_mask_path, current_date),
                           'color_mask_path': '{}{}.png'.format(save_color_mask_path, current_date)})
    else:
        form = ImageUploadForm()
    return render(request, 'image_segmentation/upload_template.html', {'form': form})


def generation_view(request):
    if request.method == 'POST':
        prompt = request.POST.get('text')
        x_coord = request.POST.get('x')
        y_coord = request.POST.get('y')
        image_path = request.POST.get('image_path')
        mask_path = request.POST.get('mask_path')

        image = Image.open(image_path)
        seg_mask = Image.open(mask_path)
        inp_mask = get_inpainting_mask(seg_mask, x_coord, y_coord)
        current_date = datetime.datetime.now().date().__str__()
        inp_mask.save('{}{}.png'.format(save_inp_mask_path, current_date))

        gen_image = get_generated_image(image, inp_mask, prompt)
        gen_image.save('{}{}.png'.format(save_gen_image_path, current_date))

        return render(request, 'image_segmentation/display_gen_template.html',
                      {'image_path': '{}{}.png'.format(save_image_path, current_date),
                       'inp_mask_path': '{}{}.png'.format(save_inp_mask_path, current_date),
                       'gen_image_path': '{}{}.png'.format(save_gen_image_path, current_date),
                       'mask_path': '{}{}.png'.format(save_mask_path, current_date)})
    else:
        return render(request, 'image_segmentation/display_gen_template.html')
