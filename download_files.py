import os
import zipfile
import shutil

import gdown


def download_and_decompress_from_drive(main_dir):
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    # Create temporary directory inside main_dir
    temp_dir = os.path.join(main_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_id = '1gZKvvjj4onwycocctaFQLtGRC-Sh2Bs8'
    d_url = f'https://drive.google.com/uc?id={file_id}'

    output = os.path.join(temp_dir, 'data.zip')
    gdown.download(d_url, output, quiet=False)

    # Decompressing the file to the temporary directory
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Move the files from temp_dir to main_dir
    for item in os.listdir(temp_dir):
        s = os.path.join(temp_dir, item)
        d = os.path.join(main_dir, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.move(s, d)

    # Delete the temporary directory
    shutil.rmtree(temp_dir)

    print(f'Successfully downloaded and decompressed files to {main_dir}.')
