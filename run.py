import cv2
from logging import getLogger, basicConfig, DEBUG, INFO
import kit_timetable_rebuilder as kitrb


def main():
    logger = getLogger(__name__)
    if __debug__:
        basicConfig(level=DEBUG)
    else:
        basicConfig(level=INFO)

    img_name = "./img/H30_jikanwari_gakubu_1zen_ページ_1.png"
    kitrb.parse(img_name)


if __name__ == '__main__':

    main()
