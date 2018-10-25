import requests
import shutil
import zipfile
import os
import logging

from .common import sha256_hash, home_directory_path, path_exists

FOLDER = '.foolbox_zoo/weights'


def fetch_weights(weights_uri, unzip=False):
    if weights_uri is None:
        logging.info(f"No weights to be fetched for this model.")
        return

    hash_digest = sha256_hash(weights_uri)
    local_path = home_directory_path(FOLDER, hash_digest)
    exists_locally = path_exists(local_path)

    filename = _filename_from_uri(weights_uri)
    file_path = os.path.join(local_path, filename)

    if exists_locally:
        logging.info(f"Weights already stored locally.")  # pragma: no cover
    else:
        _download(file_path, weights_uri, local_path)

    if unzip:
        file_path = _extract(local_path, filename)

    return file_path


def _filename_from_uri(url):
    filename = url.rsplit('/', 1)[-1]
    filename = filename.rsplit('?', 1)[0]
    return filename


def _download(file_path, url, directory):
    logging.info(f"Downloading weights: {url} to {file_path}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    # first check ETag or If-Modified-Since header or similar
    # to check whether updated weights are available?
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        raise RuntimeError(f"Failed to fetch weights from {url}")


def _extract(directory, filename):
    file_path = os.path.join(directory, filename)
    extracted_folder = filename.rsplit('.', 1)[0]
    extracted_folder = os.path.join(directory, extracted_folder)

    if not os.path.exists(extracted_folder):
        logging.info(f"Extracting weights package to {extracted_folder}")
        os.makedirs(extracted_folder)
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(extracted_folder)
        zip_ref.close()
    else:
        logging.info(f"Extraced folder already exists: {extracted_folder}")  # pragma: no cover

    return extracted_folder