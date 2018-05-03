import numpy as np
from logging import getLogger
import kit_timetable_rebuilder as kitrb


class Period(object):

    def __init__(self, image: np.ndarray, logger=None):
        self.logger = logger or getLogger(__name__)
        self.image = image

        # TODO: lessonごとにパースして保持する処理
        bin_img = kitrb.convert_to_bin(image)
        verticals = kitrb.find_side_dashed_line(bin_img, thresh=1.0)
        print("--------------------------------------------------------------------")
        print(len(verticals))
        for v in verticals:
            # print(v)
            kitrb.draw_line(image, v)
        kitrb.debug_window(image)