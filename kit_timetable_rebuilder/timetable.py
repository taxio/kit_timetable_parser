import numpy as np
from logging import getLogger


class TimeTable(object):

    def __init__(self, image: np.ndarray, logger=None):
        self.logger = logger or getLogger(__name__)

        self.image = image

        # TODO: periodごとにパースして保持する処理
