import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import shutil

MAX_IMG_ID = 23
N_VALID = 0


def get_contours(img_path: str):
    im = cv2.imread(img_path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [contour.reshape(contour.shape[0], -1).astype('int') for
                contour in contours]
    return contours


# Get all data names
names = []
for root, dirs, files in os.walk('dataset/train'):
    names += dirs
    break

# Assign id to each data name
# with open('dataset/id_to_name.txt', 'w') as f:
#     for i, name in enumerate(names):
#         # id start from 0
#         f.write(f'{i} {name}\n')
with open('dataset/train_id_to_name.txt', 'w') as f:
    for i, name in enumerate(names):
        if i <= MAX_IMG_ID - N_VALID:
            # id start from 0
            f.write(f'{i} {name}\n')
with open('dataset/valid_id_to_name.txt', 'w') as f:
    for i, name in enumerate(names):
        if i > MAX_IMG_ID - N_VALID:
            # id start from 0
            f.write(f'{i} {name}\n')

# Copy images to a image root dir and make image info
shutil.copytree('dataset/test', 'dataset/test_images', dirs_exist_ok=True)
os.makedirs('dataset/train_images', exist_ok=True)
os.makedirs('dataset/valid_images', exist_ok=True)
train_images = []
valid_images = []
for image_id, name in enumerate(names):
    im = cv2.imread(os.path.join('dataset', 'train', name,
                                 'images', f'{name}.png'))
    h, w = im.shape[0], im.shape[1]
    image_dict = {'id': image_id, 'file_name': f'{name}.png',
                  'height': h, 'width': w}
    if image_id <= MAX_IMG_ID - N_VALID:
        cv2.imwrite(os.path.join('dataset', 'train_images', f'{name}.png'),
                    im)
        train_images.append(image_dict)
    else:
        cv2.imwrite(os.path.join('dataset', 'valid_images', f'{name}.png'),
                    im)
        valid_images.append(image_dict)

# Annotations
train_annotation_dicts = []
valid_annotation_dicts = []
ann_id = 0
for img_id, name in tqdm(enumerate(names), total=len(names)):
    for root, dirs, files in os.walk(os.path.join('dataset',
                                                  'train', name, 'masks')):
        # For each mask
        for file in files:
            img_path = os.path.join(root, file)
            contours = get_contours(img_path)
            # Bbox
            min_x = np.min([np.min(contour[:, 0]) for contour in contours])
            min_y = np.min([np.min(contour[: 1]) for contour in contours])
            max_x = np.max([np.max(contour[:, 0]) for contour in contours])
            max_y = np.max([np.max(contour[:, 1]) for contour in contours])
            w = max_x - min_x
            h = max_y - min_y
            # Segmentation
            segmentation_contours = [list(map(int, list(contour.flatten()))) for
                                     contour in contours]
            # For debug plotting
            # im = cv2.imread(img_path)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # im = np.zeros_like(im)
            # im = cv2.drawContours(im, contours, -1, (255, 255, 255), 2)
            # im = cv2.rectangle(im, (min_x, min_y),
            # (min_x + w, min_y + h), (255, 0, 0), 2)
            # plt.imshow(im)
            # plt.show()
            # Make annotation
            annotation = {
                'image_id': img_id,
                'iscrowd': 0,
                'bbox': [int(min_x), int(min_y), int(w), int(h)],
                'segmentation': segmentation_contours,
                'id': ann_id,
                "category_id": 1,
            }
            ann_id += 1
            if img_id <= MAX_IMG_ID - N_VALID:
                train_annotation_dicts.append(annotation)
            else:
                valid_annotation_dicts.append(annotation)

# Make train & valid dataset json files
train_dataset = {'images': train_images,
                 'annotations': train_annotation_dicts,
                 "info": {
                     "description": "VRDL hw3",
                     "url": "",
                     "version": "1.0",
                     "year": 2021,
                     "contributor": "",
                     "date_created": ""
                 },
                 "licenses": [],
                 
                 'categories': [{'id': 1, 'name': 'nuclei'}]}
valid_dataset = {'images': valid_images,
                 'annotations': valid_annotation_dicts,
                 'categories': [{'id': 1, 'name': 'nuclei'}]}
with open('dataset/nuclei_train_dataset.json', 'w') as f:
    json.dump(train_dataset, f)
with open('dataset/nuclei_valid_dataset.json', 'w') as f:
    json.dump(valid_dataset, f)

# Make test dataset json file
test_images = []
for root, dirs, files in os.walk('dataset/test'):
    for i, file in enumerate(files):
        im = cv2.imread(os.path.join(root, file))
        h, w = im.shape[0], im.shape[1]
        image_dict = {'id': MAX_IMG_ID + i + 1,
                      'file_name': file, 'height': h, 'width': w}
        test_images.append(image_dict)
    # test_images += files
    break
test_dataset = {'images': test_images,
                'categories': [{'id': 1, 'name': 'nuclei'}]}
with open('dataset/nuclei_test_dataset.json', 'w') as f:
    json.dump(test_dataset, f)

# bb = np.zeros_like(imgray)
# plt.imshow(bb)
# plt.show()
#
# cc = cv2.drawContours(bb, a[0], -1, (255, 255, 255), 10)
# plt.imshow(cc);
# plt.show()
