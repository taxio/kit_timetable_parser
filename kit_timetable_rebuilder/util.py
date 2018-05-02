import cv2
import numpy as np


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


def calc_linear_rate(img: np.ndarray) -> float:
    """
    画像内の線の分散を返す
    :param img:
    :return:
    """
    bin_img = convert_to_bin(img, 100)
    height, width = bin_img.shape

    # 縦画像だった場合は転置
    if height > width:
        bin_img = bin_img.transpose()

    # 縦軸方向に最大値を取った1次元の行列に変換
    bin_img = bin_img.max(axis=0)

    # 線の長さを取得
    lengths = list()
    is_line = False
    buf = 0
    for b in bin_img:
        if is_line:
            if b == 255:
                buf += 1
            else:
                lengths.append(buf)
                is_line = False
                buf = 0
        else:
            if b == 255:
                is_line = True
                buf += 1
    else:
        lengths.append(buf)

    nd_lengths = np.empty(len(lengths))
    nd_lengths[:] = lengths
    var = np.var(nd_lengths)
    # self.logger.debug("linear var: {}".format(var))

    return var.item()
