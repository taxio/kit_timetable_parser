from unittest import TestCase
from nose.tools import ok_, eq_
import kit_timetable_rebuilder as kitrb
import cv2
import numpy as np


class TestUtil(TestCase):

    def setUp(self):
        self.orgn_img = cv2.imread("../img/H30_jikanwari_gakubu_1zen_ページ_1.png")

    def test_find_frames(self):
        vs = kitrb.find_vertical_frames(self.orgn_img,
                                   num_vertical=8,
                                   min_deg=0.0, max_deg=0.2)
        ss = kitrb.find_side_frames(self.orgn_img,
                               num_side=6,
                               min_deg=90.0, max_deg=90.2)
        for v in vs:
            kitrb.draw_line(self.orgn_img, v)
        for s in ss:
            kitrb.draw_line(self.orgn_img, s)

        kitrb.debug_window(self.orgn_img)

    # def test_find_vertical_frames(self):
    #     lines = kitrb.find_vertical_frames(self.orgn_img,
    #                                num_vertical=8,
    #                                min_deg=0.0, max_deg=0.2)
    #     for l in lines:
    #         print(l)
    #     eq_(True, True)
    #
    # def test_find_side_frames(self):
    #     lines = kitrb.find_side_frames(self.orgn_img,
    #                            num_side=6,
    #                            min_deg=90.0, max_deg=90.2)
    #     for l in lines:
    #         print(l)
    #     eq_(True, True)
