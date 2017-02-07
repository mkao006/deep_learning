import logging
import json
import socket
import multiprocessing
from joblib import Parallel, delayed
from functions import retrieve_images

with open('url_infos.json') as f:
    url_infos = json.load(f)


# Configure the looing
logging.basicConfig(filename='image_retrieval.log')
socket.setdefaulttimeout(30)

image_path = './images/'
# Parallel the download
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(
    delayed(retrieve_images)(url_infos=url_infos,
                             image_path=image_path,
                             group=group,
                             log_file="error.log")
    for group in url_infos.keys())
