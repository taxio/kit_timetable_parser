import cv2
import numpy as np
import math


def convert_to_bin(orgn_img: np.ndarray, bin_thresh: int= 100) -> np.ndarray:
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
        x_seg = int(rho * math.cos(rad))
        y_seg = int(rho * math.sin(rad))
        deg = rad * 180 / np.pi
        line = {
            "x_seg": x_seg,
            "y_seg": y_seg,
            "rho": rho,
            "deg": deg
        }
        lines.append(line)
    return lines


def find_lines(bin_img: np.ndarray, thresh: int, min_deg: float, max_deg: float):
    lines = cv2.HoughLines(bin_img, 1, np.pi / 180.0, thresh, np.array([]), 0, 0)
    lines = modify_hough_results(lines, bin_img.shape)
    lines = extract_at_deg(lines, min_deg=min_deg, max_deg=max_deg)
    for line in lines:
        line_lengths = calc_linear_rate(bin_img, line)
        line["lengths"] = line_lengths
        line["var"] = np.var(line_lengths)
    lines = sorted(lines, key=lambda x:x["var"])
    return lines


def find_frame_lines(bin_img: np.ndarray, num_frame: int, base_thresh: int, min_deg: float, max_deg: float) -> list:
    thresh = base_thresh
    p_delta = 0
    delta = 0
    upper_limit = num_frame * 2
    while True:
        lines = find_lines(bin_img, thresh, min_deg, max_deg)
        if lines is not None:
            print("thresh: {}, n_l: {}".format(thresh, len(lines)))
            if len(lines) > upper_limit:
                thresh += 1
                delta = 1
            elif len(lines) < num_frame:
                thresh -= 1
                delta = -1
            else:
                break
        else:
            print("thresh: {}, not found".format(thresh))
            thresh -= 1
            delta = -1

        # threshが規定値を超えたらエラー
        if thresh > max(bin_img.shape) or thresh <= 0:
            raise AssertionError

        # 範囲内に線が見つからなかったらエラー
        if p_delta and p_delta != delta:
            raise AssertionError
        p_delta = delta

    # 上位num_frame(欲しい枠線の数)だけ返す
    return lines[:min(len(lines), num_frame)]


def find_dashed_line(bin_img: np.ndarray, hough_thresh: int, var_thresh: float, min_deg: float, max_deg: float) -> list:
    """
    破線を検出する．
    hough変換に用いる閾値を少しずつ下げていき，線を検出．
    その中からvar_thresh以内を満たす破線(枠に使われている線)と思われる線を検出．
    検出した直線が増えているのに破線の数が増えない時は検出終了と見なす
    :param img:
    :param hough_thresh:
    :param var_thresh:
    :param min_deg:
    :param max_deg:
    :return:
    """
    d_line = -1000
    pn_line = 0
    d_dashed = -1000
    pn_dashed = 0
    while True:
        lines = find_lines(bin_img, thresh=hough_thresh, min_deg=min_deg, max_deg=max_deg)
        d_line = len(lines) - pn_line
        pn_line = len(lines)
        if d_line > 0:
            n_dashed = 0
            for line in lines:
                if line["var"] < var_thresh:
                    n_dashed += 1
            d_dashed = n_dashed - pn_dashed
            if d_dashed == 0:
                break
            pn_dashed = n_dashed
        hough_thresh -= 1

        # hough_threshが0までいったということは線がなかったということ
        if hough_thresh <= 0:
            return []
        # print("h_th: {}, n_line: {}, v_th: {}, n_dash: {}".format(hough_thresh, len(lines), var_thresh, pn_dashed))

    # var_thresh以内の線だけ取り出す
    lines = sorted(lines, key=lambda x:x["var"])
    dashed_lines = list()
    for line in lines:
        if line["var"] < var_thresh:
            dashed_lines.append(line)
    return dashed_lines


def find_vertical_frames(bin_img: np.ndarray, num_vertical: int, min_deg: float= 0.0, max_deg: float= 0.2) -> list:
    thresh = int(bin_img.shape[0] / 2)  # height
    verticals = find_frame_lines(bin_img, num_vertical, thresh, min_deg, max_deg)
    verticals = sorted(verticals, key=lambda x:x["x_seg"])
    return verticals


def find_side_frames(bin_img: np.ndarray, num_side: int, min_deg: float= 90.0, max_deg: float= 90.2) -> list:
    thresh = int(bin_img.shape[1] / 2)  # width
    sides = find_frame_lines(bin_img, num_side, thresh, min_deg, max_deg)
    sides = sorted(sides, key=lambda x:x["y_seg"])
    return sides


def filter_dashed_lines(dashed_lines: list, rho_thresh: int) -> list:
    selects = list()
    for line in dashed_lines:
        if int(line["rho"]) > rho_thresh:
            selects.append(line)
    return selects


def find_side_dashed_line(bin_img: np.ndarray, thresh: float, min_deg: float= 90.0, max_deg: float= 90.2) -> list:
    hough_thresh = int(min(bin_img.shape))     # 少し大きめにとる
    dashed_lines = find_dashed_line(bin_img, hough_thresh=hough_thresh, var_thresh=thresh, min_deg=min_deg, max_deg=max_deg)
    dashed_lines = filter_dashed_lines(dashed_lines, rho_thresh=5)
    return dashed_lines


def find_vertical_dashed_line(bin_img: np.ndarray, thresh: float, min_deg: float= 0.0, max_deg: float= 0.2) -> list:
    hough_thresh = int(min(bin_img.shape))     # 少し大きめにとる
    dashed_lines = find_dashed_line(bin_img, hough_thresh=hough_thresh, var_thresh=thresh, min_deg=min_deg, max_deg=max_deg)
    return dashed_lines


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


def trim_line(img: np.ndarray, line: dict, buff: int) -> np.ndarray:
    if int(line["deg"]) == 0:   # 縦線
        img = img[:, max(line["x_seg"]-buff, 0): min(line["x_seg"]+buff+1, img.shape[1])]
    elif int(line["deg"]) == 90:    # 横線
        img = img[max(line["y_seg"]-buff, 0): min(line["y_seg"]+buff+1, img.shape[0]), :]
    else:
        raise ValueError

    return img


def debug_window(img: np.ndarray):
    cv2.imshow("debug", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_line(img: np.ndarray, line: dict, color=(0, 0, 255), w=1):
    if int(line["deg"]) == 0:
        pt1 = (line["x_seg"], 0)
        pt2 = (line["x_seg"], img.shape[0])
    elif int(line["deg"]) == 90:
        pt1 = (0, line["y_seg"])
        pt2 = (img.shape[1], line["y_seg"])
    else:
        raise ValueError
    cv2.line(img, pt1, pt2, color, w, cv2.LINE_AA)


def calc_linear_rate(bin_img: np.ndarray, line: dict) -> np.ndarray:

    bin_img = trim_line(bin_img, line, buff=2)
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
        if is_line:
            lengths.append(buf)

    nd_lengths = np.empty(len(lengths))
    nd_lengths[:] = lengths

    return nd_lengths

