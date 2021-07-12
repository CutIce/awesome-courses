#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# VARIOUS FILTERS
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


def create_kernel(n):
    return [1/n/n] * n * n


def blurred(image, n):
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


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def separate_and_filter(image):
        h = image['height']
        w = image['width']
        pixels = image['pixels']

        r = [i[0] for i in pixels]
        g = [i[1] for i in pixels]
        b = [i[2] for i in pixels]

        imr = {
            'height': h,
            'width': w,
            'pixels': r
        }

        img = {
            'height': h,
            'width': w,
            'pixels': g
        }

        imb = {
            'height': h,
            'width': w,
            'pixels': b
        }

        imr_ = filt(imr)
        img_ = filt(img)
        imb_ = filt(imb)

        p = [] * len(img)

        for i in range(len(img)):
            p.append((imr_[i], img_[i], imb_[i]))

        res = {
            'height': h,
            'width': w,
            'pixels': p
        }

        return res

    return separate_and_filter


def make_blur_filter(n):

    def filter(image):
        kernel= create_kernel(n)

        res = correlate(image, kernel)

        res = round_and_clip_image(res)

        return res
    return filter


def make_sharpen_filter(n):

    def filter(image):
        kernel = create_kernel(n)

        res = correlate(image, kernel)

        res = round_and_clip_image(res)

        return res

    return filter

def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    raise NotImplementedError


# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    raise NotImplementedError


# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    raise NotImplementedError


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    raise NotImplementedError


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    raise NotImplementedError


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    raise NotImplementedError


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    raise NotImplementedError


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
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


def save_greyscale_image(image, filename, mode='PNG'):
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
    pass
