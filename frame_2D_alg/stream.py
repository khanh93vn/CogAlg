"""
Provide streaming methods to monitor some of frame_2D_alg operations.
"""

import numpy as np
import cv2 as cv

from utils import (
    blank_image, over_draw, stack_box,
    draw_blob, draw_stack,
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
        return cv.waitKey(1)

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
    sign_map = {False: BLACK, True: WHITE}  # sign_map for terminated blobs
    sign_map_unterminated = {False: DGREY, True: LGREY}  # sign_map for unterminated blobs

    def __init__(self, blob_cls, frame, winname='image_to_blobs', zoom=None):
        self.blob_cls = blob_cls
        height, width = frame['dert__'].shape[1:]
        self.box = (0, height, 0, width)
        Streamer.__init__(self, window=blank_image(self.box),
                          winname=winname,
                          zoom=zoom)
        self.incomplete_blob_ids = set()
        self.first_id = 0

    def update(self, y, P_=()):
        # draw Ps in new row
        for P in P_:
            self.img[y, P.x0 : P.x0+P.L] = self.sign_map_unterminated[P.sign]

        # add new blobs' ids, if any
        id_end = self.blob_cls.instance_cnt
        new_blobs_ids = range(self.first_id, id_end)
        self.incomplete_blob_ids.update(new_blobs_ids)
        self.first_id = id_end

        # iterate through incomplete blobs
        for blob_id in set(self.incomplete_blob_ids):
            blob = self.blob_cls.get_instance(blob_id)
            if blob is None:
                self.incomplete_blob_ids.remove(blob_id)
                continue
            elif blob.open_stacks == 0:  # terminated blob has no open_stack
                blob_box = blob.box
                self.incomplete_blob_ids.remove(blob_id)
                blob_img = draw_blob(blob, blob_box=blob_box,
                                     sign_map=Img2BlobStreamer.sign_map)
                over_draw(self.img, blob_img, blob_box)