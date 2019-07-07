"""Morphological operators that work with square shape structural elements"""
import numpy as np


def erosion(img, struct_elem):
    rows, cols = img.shape
    srows, scols = struct_elem.shape
    cr = srows // 2
    cc = scols // 2

    # padding
    left = cc
    right = scols - cc - 1
    up = cr
    down = srows - cr - 1

    tmp = np.empty_like(struct_elem, np.bool_)
    out = np.zeros_like(img, np.bool_)
    for r in range(up, rows - down):
        for c in range(left, cols - right):
            window = img[r - up: r + down + 1, c - left: c + right + 1]
            # TODO: to make it work with any struct_elem check:
            # A B
            # 1 1 1
            # 0 1 0
            # 1 0 1
            # 0 0 1
            # if B[i,j] == 0 or A[i,j] and B[i,j]
            np.logical_and(window, struct_elem, out=tmp)
            out[r, c] = tmp.all()

    return out


def dilation(img, struct_elem):
    rows, cols = img.shape
    srows, scols = struct_elem.shape
    cr = srows // 2
    cc = scols // 2

    # padding
    left = cc
    right = scols - cc - 1
    up = cr
    down = srows - cr - 1

    out = np.zeros_like(img, np.bool_)
    for r in range(up, rows - down):
        for c in range(left, cols - right):
            if img[r, c]:
                out[r - up: r + down + 1,
                    c - left: c + right + 1] = struct_elem

    return out


def closing(img, struct_elem):
    img = dilation(img, struct_elem)
    img = erosion(img, struct_elem)

    return img


def opening(img, struct_elem):
    img = erosion(img, struct_elem)
    img = dilation(img, struct_elem)

    return img


if __name__ == "__main__":
    import os
    import cv2
    import sys
    img_name = 'pr/test_imgs/erosion2.png'
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
    img = cv2.imread(img_name, 0)

    if img is None:
        print('Can not read `{}`'.format(img_name))
        exit(-1)

    img = img > 127

    cv2.imwrite('res/original.jpg', img * 255)
    os.makedirs('res', exist_ok=True)

    out = dilation(img, np.ones((5, 5), np.bool_))
    cv2.imwrite('res/dilation.jpg', out * 255)
    out = erosion(out, np.ones((5, 5), np.bool_))
    cv2.imwrite('res/erosion.jpg', out * 255)

    out = erosion(out, np.ones((40, 40), np.bool_))
    out = dilation(out, np.ones((40, 40), np.bool_))
    cv2.imwrite('res/final.jpg', out * 255)
