import numpy as np
import matplotlib.pyplot as plt

from time import time
from frame_blobs_yx import comp_pixel
from frame_blobs_seq_c_wrapper import cwrapped_derts2blobs
from utils import imread

if __name__ == "__main__":
    import argparse
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=0)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=1)
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])
    verbose = arguments['verbose']
    intra = arguments['intra']
    render = arguments['render']

    start_time = time()
    dert__ = comp_pixel(image)
    frame, idmap = cwrapped_derts2blobs(dert__)
    print(f"{len(frame.blob_)} blobs formed in {time() - start_time} seconds")

    plt.imshow(idmap, 'gray')
    plt.show()
