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
    orign_img = cv2.imread(img_name)
    parser = kitrb.TimeTableParser(orign_img)


if __name__ == '__main__':

    main()
