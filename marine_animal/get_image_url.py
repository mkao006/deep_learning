import json
from functions import get_hyponyms
from functions import get_imagenet_image_url
from nltk.corpus import wordnet as wn

# We start by taking all the subset of fish
fish_synset = wn.synsets('fish')[0]
fish_hyponym_synset = get_hyponyms(fish_synset)

# NOTE (Michael): Not sure what this is for
required_synset_name = []
required_wnid = []

# NOTE (Michael): Additional 'wnid' includes random groups that are
#                 simply used for training the model. These group will
#                 allow the model that there are non-diving related
#                 media.
additional_synset_name = []
additional_wnid = []

# Build imagenet url of each synset group
url_infos = get_imagenet_image_url(fish_hyponym_synset)

# Temporary save the url info
with open('url_infos.json', 'w') as f:
    json.dump(url_infos, f)
