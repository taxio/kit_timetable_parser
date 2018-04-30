import cv2
import numpy as np
import math

if __name__ == '__main__':

    img_name = "./img/H30_jikanwari_gakubu_1zen_ページ_1.png"
    orign_img = cv2.imread(img_name)
    gray_img = cv2.cvtColor(orign_img, cv2.COLOR_BGR2GRAY)

    print("image shape:", orign_img.shape)
    img_width = orign_img.shape[0]
    img_height = orign_img.shape[1]

    thresh = 100
    _, bin_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY_INV)

    # ここで欲しいのは
    lines = cv2.HoughLines(bin_img, 1, np.pi/180.0, 1500, np.array([]), 0, 0)

    a, b, c = lines.shape
    print(a)
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + img_height* (-b)), int(y0 + img_width * (a)))
        pt2 = (int(x0 - img_height * (-b)), int(y0 - img_width * (a)))
        cv2.line(orign_img, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("test", orign_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()