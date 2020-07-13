"""
Provide streaming methods to monitor some of frame_2D_alg operations.
"""

import numpy as np
import cv2 as cv

from utils import (
    blank_image, over_draw, draw_blob,
    BLACK, WHITE, GREY, DGREY, LGREY,
)

class Streamer:
    """
    Base class to stream visualizations.
    """
    def __init__(self, window, winname='streamer', colored=False, zoom=None):
        self.winname = winname
        self.zoom = zoom
        if isinstance(window, tuple):
            shape = (*window_shape, 3) if colored else window_shape
            self.img = np.empty(shape, 'uint8')
        else:
            self.img = window

    def update(self, **kwargs):
        """Call this method each update."""
        raise NotImplementedError

    def render(self):
        """Render visualization to screen."""
        if self.zoom is None:
            cv.imshow(winname=self.winname, mat=self.img)
        else:
            y, x = self.img.shape[:2]
            self.zoomed = cv.resize(self.img, (int(x * self.zoom),
                                               int(y * self.zoom)))
            cv.imshow(winname=self.winname,
                      mat=self.zoomed)
        cv.waitKey(1)

    def stop(self):
        cv.destroyAllWindows()

    def imwrite(self, path):
        if self.zoom is None:
            cv.imwrite(path, self.img)
        else:
            cv.imwrite(path, self.zoomed)


class Img2BlobStreamer(Streamer):
    """
    Use this class to monitor the actions of image_to_blobs in frame_blobs.
    """
    def __init__(self, frame, winname='image_to_blobs', zoom=None):
        self.frame = frame
        height, width = frame['dert__'].shape[1:]
        self.box = (0, height, 0, width)
        Streamer.__init__(self, window=blank_image(self.box),
                          winname=winname,
                          zoom=zoom)
        self.complete_blobs = []

    def update(self, y):
        for blob in self.frame['blob__']:
            if blob not in self.complete_blobs:
                if blob.open_stacks != 0:  # unterminated blob still got open_stacks
                    blob_box = blob.box[0], y + 1, *blob.box[1:]
                else:
                    blob_box = blob.box
                    self.complete_blobs.append(blob)
                # Check for newly terminated blobs
                blob_img = draw_blob(blob, blob_box=blob_box)
                over_draw(self.img, blob_img, blob_box, self.box)