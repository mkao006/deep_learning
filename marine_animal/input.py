import numpy as np
import cStringIO
import urllib
from keras.preprocessing import image
from nltk.corpus import wordnet as wn


def read_image(image_path, target_size=(224, 224)):
    # Read the image
    if image_path.startswith('http'):
        img = image.load_img(cStringIO.StringIO(
            urllib.urlopen(image_path).read()), target_size=target_size)
    else:
        img = image.load_img(image_path, target_size=target_size)
    return img


def convert_image(img):
    # Process the image
    img_array = image.img_to_array(img)
    # img_array_expanded = np.expand_dims(img_array, axis=0)
    # processed_img_array = preprocess_input(img_array_expanded)
    # return processed_img_array
    one_hot_image = img_array.flatten()
    return one_hot_image


def build_offset_dict():
    syns = list(wn.all_synsets())
    offsets_list = [(s.offset(), s) for s in syns]
    offsets_dict = dict(offsets_list)
    return offsets_dict


def convert_to_binary_label(wnid, true_offset):
    return int(int(wnid[1:]) == true_offset)


def get_all_image_path(image_dir):
    all_image_path = []
    for downloaded_wnid in os.listdir(image_dir):
        synset_path = os.path.join(image_dir, downloaded_wnid)
        for images in os.listdir(synset_path):
            all_image_path.append(os.path.join(synset_path, images))
    return all_image_path


def load_image(all_image_path, n_image=1000, offset_id):

    selected_images = np.random.choice(all_image_path, n_image)
    image_vecs = []
    label_vecs = []
    for image_path in selected_images:
        wnid = image_path.split('/')[1]
        label_vecs.append([convert_to_binary_label(wnid, offset_id)])
        image_vecs.append(convert_image(read_image(image_path)))

        image_mat = np.stack(image_vecs, axis=0)
        label_mat = np.stack(label_vecs, axis=0)
    return image_mat, label_mat
