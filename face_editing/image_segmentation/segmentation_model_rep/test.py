# import torch
# import os
# import cv2
# import numpy as np
# import torchvision.transforms as transforms
# import time
#
# from model import BiSeNet
# from PIL import Image
#
# from tools import save_mask
#
#
# def vis_parsing_maps(image, parsing_anno, stride, save_im=False, save_path=None):
#
#     part_colors = [[255, 0, 0],
#                    [255, 85, 0],
#                    [255, 170, 0],
#                    [255, 0, 85],
#                    [255, 0, 170],
#                    [0, 255, 0],
#                    [85, 255, 0],
#                    [170, 255, 0],
#                    [0, 255, 85],
#                    [0, 255, 170],
#                    [0, 0, 255],
#                    [85, 0, 255],
#                    [170, 0, 255],
#                    [0, 85, 255],
#                    [0, 170, 255],
#                    [255, 255, 0],
#                    [255, 255, 85],
#                    [255, 255, 170],
#                    [255, 0, 255],
#                    [255, 85, 255],
#                    [255, 170, 255],
#                    [0, 255, 255],
#                    [85, 255, 255],
#                    [170, 255, 255]]
#
#     image = np.array(image)
#     vis_im = image.copy().astype(np.uint8)
#     vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
#     vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
#     vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
#
#     num_of_class = np.max(vis_parsing_anno)
#
#     for pi in range(1, num_of_class + 1):
#         index = np.where(vis_parsing_anno == pi)
#         vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
#
#     vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
#     vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
#
#     # save_path = os.path.normpath(save_path)
#     save_path = './saved_images/' + time.time().__str__() + '.jpg'
#     if save_im:
#         cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno)
#         cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#
#
# def evaluate(weight_path='./res/cp', data_path='./data', weight_file_name=None):
#
#     if not os.path.exists(weight_path):
#         os.makedirs(weight_path)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     n_classes = 19
#     net = BiSeNet(n_classes=n_classes)
#     net.to(device)
#
#     save_pth = os.path.normpath(os.path.join(weight_path, weight_file_name))
#     net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
#     net.eval()
#
#     to_tensor = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     with torch.no_grad():
#         for image_path in os.listdir(data_path):
#             img = Image.open(os.path.join(data_path, image_path))
#             image = img.resize((512, 512), Image.Resampling.BILINEAR)
#             img = to_tensor(image)
#             img = torch.unsqueeze(img, 0)
#             img = img.to(device)
#             out = net(img)[0]
#             parsing = out.squeeze(0).cpu().numpy().argmax(0)
#             save_mask(parsing, '../../../mask_txt.txt')
#             vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=os.path.join(data_path, image_path))
#
#
# if __name__ == "__main__":
#     evaluate(weight_file_name='model.pth')
