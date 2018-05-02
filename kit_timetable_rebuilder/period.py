import numpy as np
from logging import getLogger


class Period(object):

    def __init__(self, image: np.ndarray, logger=None):
        self.logger = logger or getLogger(__name__)

        self.image = image

        # TODO: lessonごとにパースして保持する処理