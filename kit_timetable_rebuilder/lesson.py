import numpy as np
from logging import getLogger


class Lesson(object):

    def __init__(self, image: np.ndarray, logger=None):
        self.logger = logger or getLogger(__name__)

        self.image = image

        # TODO: 課程の情報取得や複数コマかどうかの判定をする