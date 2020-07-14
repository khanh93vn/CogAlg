from utils import *
from frame_blobs import *
import argparse

# Main #

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=0)
argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=0)
argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=0)
argument_parser.add_argument('-z', '--zoom', help='zooming ratio when rendering', type=float, default=None)
arguments = vars(argument_parser.parse_args())
image = imread(arguments['image'])
verbose = arguments['verbose']
intra = arguments['intra']
render = arguments['render']
rendering_zoom = arguments['zoom']

frame = image_to_blobs(image, verbose, render, rendering_zoom)

# Draw blobs --------------------------------------------------------------

sorted_blob_ = sorted(frame['blob__'], key=lambda blob: blob.Dert['S']) # sort by S
nblobs = len(sorted_blob_)
if nblobs < 30:
    blob_to_draw_ = sorted_blob_
else:
    blob_to_draw_ = (
        sorted_blob_[:10] +  # 10 smallest blobs
        sorted_blob_[-10:] + # 10 largest blobs
        sorted_blob_[(nblobs//2)-5 : (nblobs//2)+5]
    )

for i, blob in enumerate(blob_to_draw_):
    blob.adj_blob_ = [*zip(*blob.adj_blob_)]
    # initialize image with 3 channels (colour image)
    img_blob_ = np.zeros((frame['dert__'].shape[1], frame['dert__'].shape[2], 3))
    img_blob_box = img_blob_.copy()

    # check if there are adjacent blobs and there are unmasked values
    if blob.adj_blob_ and False in blob.dert__[0].mask:
        dert__mask = ~blob.dert__[0].mask  # get inverted mask value (we need plot mask = false)
        dert__mask = dert__mask * 255  # set intensity of colour

        # draw blobs into image
        # current blob - whilte colour
        img_blob_[blob.box[0]:blob.box[1], blob.box[2]:blob.box[3], 0] += dert__mask
        img_blob_[blob.box[0]:blob.box[1], blob.box[2]:blob.box[3], 1] += dert__mask
        img_blob_[blob.box[0]:blob.box[1], blob.box[2]:blob.box[3], 2] += dert__mask
        img_blob_box[blob.box[0]:blob.box[1], blob.box[2]:blob.box[3], 0] += dert__mask
        img_blob_box[blob.box[0]:blob.box[1], blob.box[2]:blob.box[3], 1] += dert__mask
        img_blob_box[blob.box[0]:blob.box[1], blob.box[2]:blob.box[3], 2] += dert__mask

        # draw bounding box
        cv2.rectangle(img_blob_box, (blob.box[2], blob.box[0]),
                      (blob.box[3], blob.box[1]),
                      color=(255, 255, 255), thickness=1)

        for j, adj_blob in enumerate(blob.adj_blob_[0]):

            # check if there are unmasked values
            if False in adj_blob.dert__[0].mask:
                adj_dert__mask = ~adj_blob.dert__[0].mask  # get inverted mask value (we need plot mask = false)
                adj_dert__mask = adj_dert__mask * 255  # set intensity of colour

                if blob.adj_blob_[1][j] == 1:  # external blob, colour = green
                    # draw blobs into image
                    img_blob_[adj_blob.box[0]:adj_blob.box[1], adj_blob.box[2]:adj_blob.box[3], 1] += adj_dert__mask
                    img_blob_box[adj_blob.box[0]:adj_blob.box[1], adj_blob.box[2]:adj_blob.box[3], 1] += adj_dert__mask

                    # draw bounding box
                    cv2.rectangle(img_blob_box, (adj_blob.box[2], adj_blob.box[0]),
                                  (adj_blob.box[3], adj_blob.box[1]),
                                  color=(0, 155, 0), thickness=1)

                elif blob.adj_blob_[1][j] == 0:  # internal blob, colour = red
                    # draw blobs into image
                    img_blob_[adj_blob.box[0]:adj_blob.box[1], adj_blob.box[2]:adj_blob.box[3], 2] += adj_dert__mask
                    img_blob_box[adj_blob.box[0]:adj_blob.box[1], adj_blob.box[2]:adj_blob.box[3], 2] += adj_dert__mask

                    # draw bounding box
                    cv2.rectangle(img_blob_box, (adj_blob.box[2], adj_blob.box[0]),
                                  (adj_blob.box[3], adj_blob.box[1]),
                                  color=(0, 0, 155), thickness=1)
                else:  # open， colour = blue
                    # draw blobs into image
                    img_blob_[adj_blob.box[0]:adj_blob.box[1], adj_blob.box[2]:adj_blob.box[3], 0] += adj_dert__mask
                    img_blob_box[adj_blob.box[0]:adj_blob.box[1], adj_blob.box[2]:adj_blob.box[3], 0] += adj_dert__mask

                    # draw bounding box
                    cv2.rectangle(img_blob_box, (adj_blob.box[2], adj_blob.box[0]),
                                  (adj_blob.box[3], adj_blob.box[1]),
                                  color=(155, 0, 0), thickness=1)

            else:
                break

        cv2.imwrite("./images/adj_blob_masks/mask_adj_blob_" + str(i) + ".png", img_blob_.astype('uint8'))
        # cv2.imwrite("./images/adj_blob_masks/mask_adj_blob_" + str(i) + "_box.png", img_blob_box.astype('uint8'))
