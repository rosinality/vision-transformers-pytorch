import io
import pickle

import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset


def pickle_reader(byte_str):
    return pickle.loads(byte_str)


def torch_reader(byte_str):
    return torch.load(io.BytesIO(byte_str), map_location=lambda storage, loc: storage)


def raw_reader(byte_str):
    return byte_str


def str_reader(byte_str):
    return byte_str.decode("utf-8")


class LMDBReader:
    def __init__(
        self, path, reader="torch", map_size=1024 ** 4, max_readers=126, lazy=True
    ):
        self.path = path
        self.map_size = map_size
        self.max_readers = max_readers

        self.env = None
        self.length = None

        self.reader = self.get_reader(reader)

    def open(self):
        self.env = lmdb.open(
            self.path,
            self.map_size,
            readonly=True,
            create=False,
            readahead=False,
            lock=False,
            max_readers=self.max_readers,
        )

        if not self.env:
            raise IOError(f"Cannot open lmdb dataset {self.path}")

        try:
            self.length = int(self.get(b"length", "str"))

        except KeyError:
            self.length = 0

    def get_reader(self, reader):
        if isinstance(reader, str):
            read_fn = {
                "pickle": pickle_reader,
                "torch": torch_reader,
                "raw": raw_reader,
                "str": str_reader,
            }[reader]

        elif callable(reader):
            read_fn = reader

        else:
            raise ValueError(
                'reader should be "pickle", "torch", "raw", "str" or callable'
            )

        return read_fn

    def get(self, key, reader=None):
        if self.env is None:
            self.open()

        if reader is not None:
            read_fn = self.get_reader(reader)

        else:
            read_fn = self.reader

        with self.env.begin(write=False) as txn:
            value = txn.get(key)

        if value is None:
            raise KeyError(f"lmdb dataset does not have key {key}")

        return read_fn(value)

    def __len__(self):
        if self.length is None:
            self.open()
            self.close()

        return self.length

    def __iter__(self):
        i = 0

        while i < self.length:
            yield self.__getitem__(i)
            i += 1

    def __getitem__(self, index):
        return self.get(str(index).encode("utf-8"))

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class LMDBDataset(Dataset):
    def __init__(self, path, transform):
        self.data = LMDBReader(path, reader="raw")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_bytes = self.data[index]
        class_id = int(img_bytes[:4].decode("utf-8"))
        buffer = io.BytesIO(img_bytes[4:])
        img = Image.open(buffer).convert("RGB")
        img = self.transform(img)

        return img, class_id
