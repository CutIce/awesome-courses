#!/usr/bin/env python3

import math

from PIL import Image as Image

# NO ADDITIONAL IMPORTS ALLOWED!


def get_pixel(image, x, y):
    # return image['pixels'][x, y]
    h = image['height']
    w = image['width']

    if x < 0:
        x = 0
    elif x >= w:
        x = w-1

    if y < 0:
        y = 0
    elif y >= h:
        y = h-1

    return image['pixels'][y * w + x]


def set_pixel(image, x, y, c):
    # image['pixels'][x, y] = c
    image['pixels'][y * image['width'] + x] = c


def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [0] * image['height'] * image['width'],
    }
    for y in range(image['height']):
        for x in range(image['width']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c)


# HELPER FUNCTIONS
def filter(image, x, y, kernel):
    v = 0
    h = image['height']
    w = image['width']
    size = len(kernel)

    n = int(math.sqrt(size))
    n_ = int(n // 2)

    for i in range(size):
        x1 = i // n
        y1 = i % n
        x_ = x1 - n_
        y_ = y1 - n_
        v += kernel[(y_ + n_) * n + (x_ + n_)] * get_pixel(image, x+x_, y+y_)

    return v

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [0] * image['height'] * image['width']
    }

    h = image['height']
    w = image['width']

    for x in range(w):
        for y in range(h):
            result['pixels'][y * w + x] = filter(image, x, y, kernel)

    return result


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    result = image.copy()

    s = len(result['pixels'])

    for i in range(s):
        x = round(result['pixels'][i])
        if x < 0:
            x = 0
        elif x > 255:
            x = 255

        result['pixels'][i] = x

    return result


# FILTERS
def create_kernel(n):
    return [1/n/n] * n * n


def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = create_kernel(n)

    # print(kernel)

    # then compute the correlation of the input image with that kernel
    correlation = correlate(image, kernel)

    # print(correlation)
    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    result = round_and_clip_image(correlation)

    return result


def sharpened(image, n):
    kernel = create_kernel(n)
    blur = correlate(image, kernel)

    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [0] * image['height'] * image['width'],
    }

    for i in range(len(result['pixels'])):
        result['pixels'][i] = round(2 * image['pixels'][i] - blur['pixels'][i])

    result = round_and_clip_image(result)

    return result


def edges(image):
    kx = [-1, 0, 1,
          -2, 0, 2,
          -1, 0, 1]
    ky = [-1, -2, -1,
           0,  0,  0,
           1,  2,  1]
    
    ox = correlate(image, kx)
    oy = correlate(image, ky)
    
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [0] * len(image['pixels'])
    }
    
    for i in range(len(result['pixels'])):
        result['pixels'][i] = math.sqrt(ox['pixels'][i] ** 2 + oy['pixels'][i] ** 2)

    result = round_and_clip_image(result)
    
    return result

# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    # ori = load_image('test_images', 'centered_pixel.png')
    # res = inverted(ori)
    # print(res)

    origin = {
        'height': 5,
        'width': 5,
        'pixels': [0,  1,  2,  3,  4,
                   5,  6,  7,  8,  9,
                   10, 11, 12, 13, 14,
                   15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24]
    }

    res = blurred(origin, 3)

    print(res['pixels'])

    print('------------------------------------------------------------')

    origin1 = {
        'height': 5,
        'width': 5,
        'pixels': [0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0,
                   0,  0,  0,  0,  0]
    }

    res1 = blurred(origin1, 3)
    print(res1['pixels'])
    
    print('------------------------------------------------------------')

    origin2 = {
        'height': 5,
        'width': 5,
        'pixels': [1,  1,  1,  1,  1,
                   1,  1,  1,  1,  1,
                   1,  1,  1,  1,  1,
                   1,  1,  1,  1,  1,
                   1,  1,  1,  1,  1]
    }

    res2 = blurred(origin2, 3)
    print(res2['pixels'])
