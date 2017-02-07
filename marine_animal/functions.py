import os
import urllib
import imghdr


# NOTE (Michael): Required 'wnid' are groups of keywords that are of
#                 interest. For example, sharks and rays.


def get_hyponyms(synset):
    ''' This function retrieves all the hyponym of a synset.
    '''
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())


def build_imagenet_download_url(wnid):
    '''This function builds the imagenet url for a wnid where individual
    urls can be obtains.

    '''

    return 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={0}'.format(wnid)


def offset_to_wnid(offset):
    return 'n' + str(offset()).zfill(8)


def wnid_to_offset(wnid):
    return str(int(wnid[1:]))


def get_imagenet_image_url(synset):
    ''' Function to get the individual image url from imagenet.
    '''

    selected_wnid = [offset_to_wnid(sset.offset())
                     for sset in synset]
    selected_name = [sset.name()
                     for sset in synset]
    selected_url = [build_imagenet_download_url(wnid)
                    for wnid in selected_wnid]
    info = [{'wnid': wnid, 'synset_name': name, 'url': url}
            for wnid, name, url in
            zip(selected_wnid, selected_name, selected_url)]

    url_infos = {}
    for entry in info:
        current_wnid = entry.get('wnid')
        current_name = entry.get('synset_name')
        print('Extracting image url for (wnid: {0}, name: {1})'.format(
            current_wnid, current_name))

        image_urls = urllib.urlopen(entry.get('url')).read().split()
        url_infos[current_wnid] = image_urls
    return url_infos


def retrieve_images(url_infos, image_path, group, log_file):
    '''Retrieve the images from imagenet.
    '''

    file_destination = os.path.join(image_path, group)
    try:
        os.stat(file_destination)
    except:
        os.mkdir(file_destination)
        for index, image_url in enumerate(url_infos[group]):
            print(image_url)
            try:
                image_destination = '{0}/image{1}.jpg'.format(
                    file_destination, index)
                urllib.urlretrieve(image_url, image_destination)
                image_filetype = imghdr.what(image_destination)
                if image_filetype not in set(['gif', 'jpeg', 'png']):
                    os.remove(image_destination)
            except:
                msg = 'Image({0}, {1}) retrieval failed.\n'.format(
                    group, image_url.encode('utf-8'))
                with open(log_file, 'a+') as f:
                    f.write(msg)
