import numpy as np
from logging import getLogger


class Lesson(object):

    def __init__(self, image: np.ndarray, name=None, logger=None):
        self.logger = logger or getLogger(__name__)

        self.image = image