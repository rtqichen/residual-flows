"""
Run the following commands in ~ before running this file
wget http://image-net.org/small/train_64x64.tar
wget http://image-net.org/small/valid_64x64.tar
tar -xvf train_64x64.tar
tar -xvf valid_64x64.tar
wget http://image-net.org/small/train_32x32.tar
wget http://image-net.org/small/valid_32x32.tar
tar -xvf train_32x32.tar
tar -xvf valid_32x32.tar
"""

import numpy as np
import scipy.ndimage
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def convert_path_to_npy(*, path='train_64x64', outfile='train_64x64.npy'):
    assert isinstance(path, str), "Expected a string input for the path"
    assert os.path.exists(path), "Input path doesn't exist"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    print('Number of valid images is:', len(files))
    imgs = []
    for i in tqdm(range(len(files))):
        img = scipy.ndimage.imread(join(path, files[i]))
        img = img.astype('uint8')
        assert np.max(img) <= 255
        assert np.min(img) >= 0
        assert img.dtype == 'uint8'
        assert isinstance(img, np.ndarray)
        imgs.append(img)
    resolution_x, resolution_y = img.shape[0], img.shape[1]
    imgs = np.asarray(imgs).astype('uint8')
    assert imgs.shape[1:] == (resolution_x, resolution_y, 3)
    assert np.max(imgs) <= 255
    assert np.min(imgs) >= 0
    print('Total number of images is:', imgs.shape[0])
    print('All assertions done, dumping into npy file')
    np.save(outfile, imgs)


if __name__ == '__main__':
    convert_path_to_npy(path='train_64x64', outfile='train_64x64.npy')
    convert_path_to_npy(path='valid_64x64', outfile='valid_64x64.npy')
    convert_path_to_npy(path='train_32x32', outfile='train_32x32.npy')
    convert_path_to_npy(path='valid_32x32', outfile='valid_32x32.npy')
