import os
import re
from PIL import Image


# 1. Load the group image
#
# 2. Resize images
#

target_size = (1024, 512)

image_path = './images/'
target_image_path = './processed_images/'
if not os.path.exists(target_image_path):
    os.mkdir(target_image_path)
image_names = [os.path.join(root, name)
               for root, dirs, files in os.walk(image_path)
               for name in files]

for image in image_names:
    img = Image.open(image)
    resized_image = img.resize(target_size)
    group_name = image.split('/')[-2]
    image_name = image.split('/')[-1]
    new_image_dir = os.path.join(target_image_path, group_name)
    new_image_path = os.path.join(new_image_dir, image_name)
    if not os.path.exists(new_image_dir):
        os.mkdir(new_image_dir)
    try:
        resized_image.save(new_image_path)
    except IOError as e:
        if e.message != 'cannot write mode P as JPEG':
            raise
        else:
            resized_image.convert('RGB').save(new_image_path)
