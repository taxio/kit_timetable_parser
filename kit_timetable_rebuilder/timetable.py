import numpy as np
from logging import getLogger
import kit_timetable_rebuilder as kitrb


class TimeTable(object):

    def __init__(self, image: np.ndarray, logger=None):
        self.logger = logger or getLogger(__name__)
        self.image = image

        # 枠線検出
        verticals = kitrb.find_vertical_frames(image,
                                               num_vertical=8,
                                               min_deg=0.0,
                                               max_deg=0.2)
        sides = kitrb.find_side_frames(image,
                                       num_side=6,
                                       min_deg=90.0,
                                       max_deg=90.2)

        # 枠線に沿ってトリミング, Periodオブジェクトとして保持
        periods = list()
        for d in range(3):
            days = list()
            for p in range(6):
                dst_img = image[
                          sides[d + 2]["y_seg"]:sides[d + 3]["y_seg"],
                          verticals[p + 1]["x_seg"]:verticals[p + 2]["x_seg"]
                          ]
                period = kitrb.Period(dst_img)
                days.append(period)
            periods.append(days)
