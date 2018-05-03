import cv2
import numpy as np
import math
from logging import getLogger
import kit_timetable_rebuilder as kitrb


def parse(image_name: str):
    orgn_img = cv2.imread(image_name)
    timetable = kitrb.TimeTable(orgn_img)
