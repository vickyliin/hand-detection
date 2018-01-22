import numpy as np
from skimage import img_as_float
from skimage.draw import line
from skimage.io import imread, imsave

def draw_rectangle(img, x0, y0, x1, y1, color=(0, 0, 0)):
    draw_line(img, x0, y0, x0, y1, color)
    draw_line(img, x0, y0, x1, y0, color)
    draw_line(img, x1, y0, x1, y1, color)
    draw_line(img, x0, y1, x1, y1, color)

def draw_line(img, x0, y0, x1, y1, color=(0, 0, 0)):
    x0 = min(img.shape[1]-1, max(0, x0))
    y0 = min(img.shape[0]-1, max(0, y0))
    x1 = min(img.shape[1]-1, max(0, x1))
    y1 = min(img.shape[0]-1, max(0, y1))
    yy, xx = line(y0,x0,y1,x1)
    img[yy, xx, :] = color

def draw_bbox(img, boxes, colors=[(255, 0), (0, 255)]):
    img = img.copy()
    for c, ((x0, y0, x1, y1), p) in boxes.items():
        draw_rectangle(img, x0+2, y0+2, x1+2, y1+2, color=(0, 0, 0))
        color = colors[c]
        color += (int(p * 255), )
        draw_rectangle(img, x0, y0, x1, y1, color=color)
        draw_rectangle(img, x0-1, y0-1, x1+1, y1+1, color=color)
        draw_rectangle(img, x0+1, y0+1, x1-1, y1-1, color=color)
    return img

