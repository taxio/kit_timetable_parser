import cv2
import numpy as np
import math


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


def modify_hough_results(hough_results: list, img_size: tuple) -> list:
    """
    cv2.HoughLinesの結果を整形する．
    :param hough_results: ハフ変換結果のリスト
    :param img_size: ライン検出に使った画像のサイズ(height, width)
    :return: 各線の情報を整形したリスト
    """
    n_lines, _, _ = hough_results.shape
    height, width = img_size
    lines = list()
    for i in range(n_lines):
        rho = hough_results[i][0][0]
        rad = hough_results[i][0][1]
        a = math.cos(rad)
        b = math.cos(rad)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + width * (-b)), int(y0 + height * (a)))
        pt2 = (int(x0 - width * (-b)), int(y0 - height * (a)))
        deg = rad * 180 / np.pi
        line = {
            "pt1": pt1,
            "pt2": pt2,
            "rho": rho,
            "deg": deg
        }
        lines.append(line)
    return lines


def find_frame_lines(bin_img: np.ndarray, num_frame: int, base_thresh: int, min_deg: float, max_deg: float) -> list:
    thresh = base_thresh
    p_delta = 0
    delta = 0
    while True:
        lines = cv2.HoughLines(bin_img, 1, np.pi / 180.0, thresh, np.array([]), 0, 0)
        if lines is not None:
            lines = modify_hough_results(lines, bin_img.shape)
            lines = extract_at_deg(lines, min_deg=min_deg, max_deg=max_deg)
            print("thresh: {}, n_f: {}".format(thresh, len(lines)))
            if len(lines) < num_frame:
                thresh -= 1
                delta = -1
            elif len(lines) > num_frame:
                thresh += 1
                delta = 1
            else:
                break
        else:
            print("thresh: {}, not found".format(thresh))
            thresh += 1
            delta = 1

        # threshが規定値を超えたらエラー
        if thresh > max(bin_img.shape) or thresh <= 0:
            raise AssertionError

        # 値がピンポイントで見つからなかったらエラー
        # TODO: 改善する
        if p_delta and p_delta != delta:
            raise AssertionError
        p_delta = delta

    return lines


def find_vertical_frames(img: np.ndarray, num_vertical: int, min_deg: float, max_deg: float) -> list:
    bin_img = convert_to_bin(img, 100)
    thresh = int(bin_img.shape[0] / 2)  # height
    verticals = find_frame_lines(bin_img, num_vertical, thresh, min_deg, max_deg)
    return verticals


def find_side_frames(img: np.ndarray, num_side: int, min_deg: float, max_deg: float) -> list:
    bin_img = convert_to_bin(img, 100)
    thresh = int(bin_img.shape[1] / 2)  # width
    sides = find_frame_lines(bin_img, num_side, thresh, min_deg, max_deg)
    return sides


def extract_at_deg(lines: list, min_deg: float, max_deg: float) -> list:
    """
    指定の角度に収まる線だけ取り出す
    :param lines: 整形済みの線のリスト
    :param min_deg: 下限
    :param max_deg: 上限
    :return: 指定の角度以内の線のリスト
    """
    selects = list()
    for line in lines:
        if min_deg <= line["deg"] <= max_deg:
            selects.append(line)
    return selects


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
