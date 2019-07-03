import numpy as np
from collections import deque


def con_component(img):
    rows, cols = img.shape
    label = 1
    queue = deque()
    labels = np.zeros_like(img, dtype=int)

    for i in range(rows):
        for j in range(cols):
            if _check_pixel((i, j), img, labels):
                labels[i, j] = label
                queue.append((i, j))
            else:
                continue

            while len(queue):
                pos = queue.popleft()
                r, c = pos
                neighbors = []

                top = r - 1 > 0
                bot = r + 1 < rows
                left = c - 1 > 0
                right = c + 1 < cols

                if top:
                    neighbors.append((r - 1, c))

                if bot:
                    neighbors.append((r + 1, c))

                if left:
                    neighbors.append((r, c - 1))

                if right:
                    neighbors.append((r, c+1))

                if top and left:
                    neighbors.append((r - 1, c - 1))

                if top and right:
                    neighbors.append((r - 1, c + 1))

                if bot and left:
                    neighbors.append((r + 1, c - 1))

                if bot and right:
                    neighbors.append((r + 1, c + 1))

                for npos in neighbors:
                    if _check_pixel(npos, img, labels):
                        labels[npos] = label
                        queue.append(npos)

            label += 1

    return labels


def _check_pixel(pos, img, labels):
    """Check if pixel is foreground and doesnt has not label"""
    return img[pos] and not labels[pos]


def color_regions(labels):
    rows, cols = labels.shape

    out = np.empty((rows, cols, 3))

    for i in range(rows):
        for j in range(cols):
            lab = labels[i, j]
            if not lab:
                continue

            color = lab % 3

            if color == 0:
                out[i, j, 0] = 255
            if color == 1:
                out[i, j, 1] = 255

            if color == 2:
                out[i, j, 2] = 255

    return out


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

    img = img > 60

    cv2.imwrite('res/original.jpg', img * 255)

    out = con_component(img)

    colors = color_regions(out)

    cv2.imwrite('res/regions.png', colors)

    img = out == 1

    from morphological import dilation, erosion

    out = dilation(img, np.ones((3, 3), dtype=bool))
    cv2.imwrite('res/dilation.jpg', out * 255)
    out = erosion(out, np.ones((3, 3), dtype=bool))
    cv2.imwrite('res/erosion.jpg', out * 255)

    out = erosion(out, np.ones((40, 40), np.bool_))
    out = dilation(out, np.ones((40, 40), np.bool_))
    cv2.imwrite('res/final.jpg', out * 255)

