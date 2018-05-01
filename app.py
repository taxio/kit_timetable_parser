import cv2
import numpy as np
from timetable_parser import TimeTableParser


if __name__ == '__main__':

    img_name = "./img/H30_jikanwari_gakubu_1zen_ページ_1.png"
    orign_img = cv2.imread(img_name)
    parser = TimeTableParser(orign_img)
