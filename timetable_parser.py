import cv2
import numpy as np
import math
from logging import getLogger, StreamHandler


class TimeTableParser:

    def __init__(self, image: np.ndarray, logger=None):
        # logger setting
        self.logger = logger or getLogger(__name__)

        # 時間割から各授業のオブジェクトを生成
        self.orgn_img = image
        self.bin_img = self.convert_to_bin(self.orgn_img, 100)
        self.width, self.height = self.bin_img.shape
        lines = self.find_lines()
        verticals, sides = self.filter_frame_lines(lines)
        days_periods = self.trim_periods(verticals, sides)

    @staticmethod
    def convert_to_bin(orgn_img: np.ndarray, bin_thresh: int) -> np.ndarray:
        """
        カラー画像から2値画像に変換
        :param orgn_img: カラー画像
        :param bin_thresh: 2値化する際の閾値(今回はそもそも白黒画像なので適当でいい)
        :return: 2値化画像
        """
        gray_img = cv2.cvtColor(orgn_img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray_img, bin_thresh, 255, cv2.THRESH_BINARY_INV)
        return bin_img

    def find_lines(self) -> list:
        """
        ハフ変換で画像内の線を見つける．
        画像内で使われているのは縦横合わせて14本なので，大体20~24本見つけるまで閾値を調整し続ける
        :return: 線のリスト
        """
        line_thresh = int(min(self.width, self.height)/2)
        self.logger.info("finding lines...")
        while True:
            lines = cv2.HoughLines(self.bin_img, 1, np.pi / 180.0, line_thresh, np.array([]), 0, 0)
            if lines is not None:
                n_lines, _, _ = lines.shape
                if n_lines < 20:
                    line_thresh += 10
                elif n_lines > 28:
                    line_thresh -= 10
                else:
                    self.logger.info("found")
                    self.logger.debug("num of lines: {}".format(n_lines))
                    break

        return lines

    def filter_frame_lines(self, lines: list) -> tuple:
        """
        ハフ変換で得られた線群からフレームに使われている縦横線を見つける
        :param lines: 線群
        :return: 縦線のリスト, 横線のリスト
        """
        n_lines, _, _ = lines.shape
        self.logger.info("finding frame lines...")
        cands = list()
        for i in range(n_lines):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + self.height * (-b)), int(y0 + self.width * (a)))
            pt2 = (int(x0 - self.height * (-b)), int(y0 - self.width * (a)))
            theta = int(theta*180/np.pi)
            direction = 1   # 1:縦線 -1:横線
            if theta in range(89, 92):  # 横線(89~91度以内)
                dst_img = self.bin_img[pt1[1]-2:pt2[1]+2, 0:]
                direction = -1
            elif theta in range(-1, 2): # 縦線(-1~1度以内)
                dst_img = self.bin_img[0:, pt1[0]-2:pt2[0]+2]
            else:
                continue
            avg = np.average(dst_img)
            cand = {
                "pt1": pt1,
                "pt2": pt2,
                "avg": avg,
                "direction": direction
            }
            cands.append(cand)

        avg_thresh = 10
        while True:
            frames = list()
            verticals = list()
            sides = list()
            for cand in cands:
                if cand['avg'] > avg_thresh:
                    if cand['direction'] == 1:
                        verticals.append(cand)
                    else:
                        sides.append(cand)

            if (len(verticals)+len(sides)) == 14:
                if len(verticals) == 8 and len(sides) == 6:
                    break
                else:
                    self.logger.error("frame lines cannot found")
                    exit(1)

            if (len(verticals)+len(sides)) < 14:
                avg_thresh -= 1
            else:
                avg_thresh += 1

        self.logger.info("found")

        # 縦線，横線ごとにソート
        self.logger.debug("verticals")
        verticals = sorted(verticals, key=lambda x: x['pt1'][0])
        for v in verticals:
            self.logger.debug("{}, {}".format(v['pt1'], v['pt2']))

        self.logger.debug("sides")
        sides = sorted(sides, key=lambda x: x['pt1'][1])
        for s in sides:
            self.logger.debug("{}, {}".format(s['pt1'], s['pt2']))

        return verticals, sides

    def trim_periods(self, verticals: list, sides: list) -> list:
        """
        各フレームの縦横線から曜日・時間ごとの授業群を元画像からトリミング
        :param verticals: 縦線のリスト
        :param sides: 横線のリスト
        :return: [曜日[時間ごとの授業群]]のリスト
        """
        days = list()
        for w in range(3):
            periods = list()
            for i in range(6):
                dst_img = self.orgn_img[sides[w + 2]['pt1'][1]:sides[w + 3]['pt1'][1],
                          verticals[i + 1]['pt1'][0]:verticals[i + 2]['pt1'][0]]
                periods.append(dst_img)
            days.append(periods)

        return days


