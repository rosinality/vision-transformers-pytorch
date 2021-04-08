import sys

import lmdb
from tqdm import tqdm
from torchvision.datasets import DatasetFolder


def file_read(filename):
    with open(filename, 'rb') as f:
        return f.read()


if __name__ == '__main__':
    root = sys.argv[1]
    name = sys.argv[2]

    IMG_EXTENSIONS = (
        '.jpg',
        '.jpeg',
        '.png',
        '.ppm',
        '.bmp',
        '.pgm',
        '.tif',
        '.tiff',
        '.webp',
    )

    dset = DatasetFolder(root, file_read, IMG_EXTENSIONS)

    with lmdb.open(f'{name}.lmdb', map_size=1024 ** 4, readahead=False) as env:
        for i in tqdm(range(len(dset))):
            img, class_id = dset[i]
            class_byte = str(class_id).zfill(4).encode('utf-8')

            with env.begin(write=True) as txn:
                txn.put(str(i).encode('utf-8'), class_byte + img)

        with env.begin(write=True) as txn:
            txn.put(b'length', str(len(dset)).encode('utf-8'))
