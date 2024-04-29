from ultralytics import YOLO
import torch

# Data preprocessing
from tqdm.auto import tqdm
import os
import requests
import zipfile
import cv2
import math
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
from PIL import Image


# ROOT_DIR = '/Users/Kian/Documents/VU AI/Thesis/Ultralytics GitHub/ultralytics-main/hit-uav'
# train_imgs_dir = 'train/images'
# train_labels_dir = 'train/labels'
# val_imgs_dir = 'val/images'
# val_labels_dir = 'val/labels'
# test_imgs_dir = 'test/images'
# test_labels_dir = 'test/labels'
# classes = ['Person', 'Car', 'Bicycle', 'OtherVechicle', 'DontCare']
# colors = np.random.uniform(0, 255, size=(len(classes), 3))



# # Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
# def yolo2bbox(bboxes):
#     xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
#     xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
#     return xmin, ymin, xmax, ymax



# def plot_box(image, bboxes, labels, classes=classes, colors=colors, pos='above'):
#     # Need the image height and width to denormalize
#     # the bounding box coordinates
#     height, width, _ = image.shape
#     lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
#     tf = max(lw - 1, 1) # Font thickness.
#     for box_num, box in enumerate(bboxes):
#         x1, y1, x2, y2 = yolo2bbox(box)
#         # denormalize the coordinates
#         xmin = int(x1*width)
#         ymin = int(y1*height)
#         xmax = int(x2*width)
#         ymax = int(y2*height)

#         p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

#         class_name = classes[int(labels[box_num])]

#         color=colors[classes.index(class_name)]

#         cv2.rectangle(
#             image,
#             p1, p2,
#             color=color,
#             thickness=lw,
#             lineType=cv2.LINE_AA
#         )

#         # For filled rectangle.
#         w, h = cv2.getTextSize(
#             class_name,
#             0,
#             fontScale=lw / 3,
#             thickness=tf
#         )[0]

#         outside = p1[1] - h >= 3

#         if pos == 'above':
#             p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#             cv2.rectangle(
#                 image,
#                 p1, p2,
#                 color=color,
#                 thickness=-1,
#                 lineType=cv2.LINE_AA
#             )
#             cv2.putText(
#                 image,
#                 class_name,
#                 (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=lw/3.5,
#                 color=(255, 255, 255),
#                 thickness=tf,
#                 lineType=cv2.LINE_AA
#             )
#         else:
#             new_p2 = p1[0] + w, p2[1] + h + 3 if outside else p2[1] - h - 3
#             cv2.rectangle(
#                 image,
#                 (p1[0], p2[1]), new_p2,
#                 color=color,
#                 thickness=-1,
#                 lineType=cv2.LINE_AA
#             )
#             cv2.putText(
#                 image,
#                 class_name,
#                 (p1[0], p2[1] + h + 2 if outside else p2[1]),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=lw/3,
#                 color=(255, 255, 255),
#                 thickness=tf,
#                 lineType=cv2.LINE_AA
#             )
#     return image

# print('Succes data pre')


# #   Function to plot images with the bounding boxes.
# def plot(image_path, label_path, num_samples, classes=classes, colors=colors, pos='above'):
#     all_training_images = glob.glob(image_path+'/*')
#     all_training_labels = glob.glob(label_path+'/*')
#     all_training_images.sort()
#     all_training_labels.sort()

#     temp = list(zip(all_training_images, all_training_labels))
#     random.shuffle(temp)
#     all_training_images, all_training_labels = zip(*temp)
#     all_training_images, all_training_labels = list(all_training_images), list(all_training_labels)

#     num_images = len(all_training_images)

#     if num_samples == -1:
#         num_samples = num_images

#     num_cols = 2
#     num_rows = int(math.ceil(num_samples / num_cols))

#     plt.figure(figsize=(10 * num_cols, 6 * num_rows))
#     for i in range(num_samples):
#         image_name = all_training_images[i].split(os.path.sep)[-1]
#         image = cv2.imread(all_training_images[i])
#         with open(all_training_labels[i], 'r') as f:
#             bboxes = []
#             labels = []
#             label_lines = f.readlines()
#             for label_line in label_lines:
#                 label, x_c, y_c, w, h = label_line.split(' ')
#                 x_c = float(x_c)
#                 y_c = float(y_c)
#                 w = float(w)
#                 h = float(h)
#                 bboxes.append([x_c, y_c, w, h])
#                 labels.append(label)
#         result_image = plot_box(image, bboxes, labels, classes, colors, pos)
#         plt.subplot(num_rows, num_cols, i+1) # Visualize 2x2 grid of images.
#         plt.imshow(image[:, :, ::-1])
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()




# # Visualize a few training images.
# plot(
#     image_path=os.path.join(ROOT_DIR, train_imgs_dir),
#     label_path=os.path.join(ROOT_DIR, train_labels_dir),
#     num_samples=4
# )







# ------------ SELECT PRE-TRAINED MODEL ------------
pretrained_model = YOLO('SteStu_hitUAV_50e.pt')
# pretrained_model = YOLO('yolov8n.pt')
modified_model = YOLO('yolov8n.yaml') 



def load_pretrained_params_by_size(pretrained_model, modified_model):
    # Get the state dictionaries of the models
    pretrained_state_dict = pretrained_model.state_dict()
    modified_state_dict = modified_model.state_dict()

    # Create a new state dictionary for the modified model
    new_state_dict = {}

    # Iterate through the parameters of the modified model
    for modified_name, modified_param in modified_state_dict.items():
        # Skip loading into specific layers (e.g., layers named 'ssf')
        if 'ssf' in modified_name:
            print(f"Skipping loading pretrained parameter into '{modified_name}'")
            continue
        
        # Try to find a compatible pretrained parameter by size
        found_match = False
        for pretrained_name, pretrained_param in pretrained_state_dict.items():
            if pretrained_param.shape == modified_param.shape:
                new_state_dict[modified_name] = pretrained_param
                print(f"Loaded pretrained parameter into '{modified_name}' based on size match")
                found_match = True
                break
        
        if not found_match:
            print(f"No compatible pretrained parameter found for '{modified_name}'")

    # Load the new state dictionary into the modified model
    modified_model.load_state_dict(new_state_dict, strict=False)
    return modified_model



def freeze_model_except_ssf(modified_model):
    # Iterate through the parameters of the modified model
    for name, param in modified_model.named_parameters():
        # Check if the parameter belongs to a layer containing 'ssf'
        if 'ssf' in name:
            # Unfreeze parameters in layers containing 'ssf'
            param.requires_grad = True
            print(f"Unfreezing parameter: {name}")
        else:
            # Freeze parameters in other layers
            param.requires_grad = False
            print(f"Freezing parameter: {name}")

    return modified_model  # Return the modified model after freezing



# Load pretrained parameters into the modified model based on size compatibility
modified_model = load_pretrained_params_by_size(pretrained_model, modified_model)

# Freeze all weights except for parameters in 'ssf' layers
modified_model = freeze_model_except_ssf(modified_model)


# Now you can use the modified_model for further training or evaluation


# Assuming modified_model is your PyTorch model that you want to save
torch.save(modified_model, '/Users/Kian/Documents/VU AI/Thesis/Models/SSF_model/modifiedModel/modified_model.pt')
# Now upload 'modified_model.pth' to Google Colab



print('====================')

# print('--------Start Training--------')



# Train the model
modified_trained = modified_model.train(
   data='/Users/Kian/Documents/VU AI/Thesis/Ultralytics GitHub/ultralytics-main/hit-uav/dataset.yaml',
   imgsz=512,
   epochs=1,
   batch=16,
   name='modified_trained'
)

print(modified_model)

print('--------Done Training--------')



# results = modified_trained(['/Users/Kian/Documents/VU AI/Thesis/Ultralytics GitHub/ultralytics-main/hit-uav/test/images/1_80_60_0_08686.jpg'])  # results list

# # results = pretrained_model(['/Users/Kian/Documents/VU AI/Thesis/Ultralytics GitHub/ultralytics-main/hit-uav/test/images/1_80_60_0_08686.jpg'])  # results list


# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen

#     result.save(filename='testResult.jpg')  # save to disk






# params = ['ssf',
# 'model.37.cv2.0.0.conv.weight',
# 'model.37.cv2.0.0.bn.bias',
# 'model.37.cv2.0.1.conv.weight',
# 'model.37.cv2.0.1.bn.weight',
# 'model.37.cv2.0.1.bn.bias',
# 'model.37.cv2.0.2.weight',
# 'model.37.cv2.0.2.bias',
# 'model.37.cv2.1.0.conv.weight ',
# 'model.37.cv2.1.0.bn.weight',
# 'model.37.cv2.1.0.bn.bias',
# 'model.37.cv2.1.1.conv.weight',
# 'model.37.cv2.1.1.bn.weight',
# 'model.37.cv2.1.1.bn.bias',
# 'model.37.cv2.1.2.weight',
# 'model.37.cv2.1.2.bias ',
# 'model.37.cv2.2.0.conv.weight',
# 'model.37.cv2.2.0.bn.weight',
# 'model.37.cv2.2.0.bn.bias',
# 'model.37.cv2.2.1.conv.weight',
# 'model.37.cv2.2.1.bn.weight',
# 'model.37.cv2.2.1.bn.bias ',
# 'model.37.cv2.2.2.weight',
# 'model.37.cv2.2.2.bias',
# 'model.37.cv3.0.0.conv.weight',
# 'model.37.cv3.0.0.bn.weight',
# 'model.37.cv3.0.0.bn.bias',
# 'model.37.cv3.0.1.conv.weight',
# 'model.37.cv3.0.1.bn.weight',
# 'model.37.cv3.0.1.bn.bias',
# 'model.37.cv3.0.2.weight',
# 'model.37.cv3.0.2.bias',
# 'model.37.cv3.1.0.conv.weight',
# 'model.37.cv3.1.0.bn.weight',
# 'model.37.cv3.1.0.bn.bias',
# 'model.37.cv3.1.1.conv.weight',
# 'model.37.cv3.1.1.bn.weight',
# 'model.37.cv3.1.1.bn.bias',
# 'model.37.cv3.1.2.weight',
# 'model.37.cv3.1.2.bias',
# 'model.37.cv3.2.0.conv.weight',
# 'model.37.cv3.2.0.bn.weight',
# 'model.37.cv3.2.0.bn.bias',
# 'model.37.cv3.2.1.conv.weight',
# 'model.37.cv3.2.1.bn.weight',
# 'model.37.cv3.2.1.bn.bias',
# 'model.37.cv3.2.2.weight',
# 'model.37.cv3.2.2.bias',
# 'model.37.dfl.conv.weight',]