import cv2
from logging import getLogger, basicConfig, DEBUG, INFO
from kit_timetable_rebuilder import TimeTableParser


def main():
    logger = getLogger(__name__)
    if __debug__:
        basicConfig(level=DEBUG)
    else:
        basicConfig(level=INFO)

    img_name = "./img/H30_jikanwari_gakubu_1zen_ページ_1.png"
    orign_img = cv2.imread(img_name)
    parser = TimeTableParser(orign_img)


if __name__ == '__main__':

    main()
