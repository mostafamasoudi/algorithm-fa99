from argparse import ArgumentParser
from PIL import Image
import numpy


def get_args():
    """
        read command line arguments with this positional args
        input_file
        dx
        dy
        like this:
        >>> python main.py input_file dx dy
    """
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("dx", type=int)
    parser.add_argument("dy", type=int)
    args = parser.parse_args()
    return args


def read_image(input_file):
    ''' read input image '''
    image = Image.open(input_file)
    rgb_image = image.convert('RGB')
    return rgb_image


def write_image(image_array, path):
    image_array = image_array.astype(numpy.uint8)
    new_im = Image.fromarray(image_array)
    new_im.save(path)


def coloring_seams(im_array, mask):
    """ make seams white and save current frame for create gif """
    current_frame = im_array.copy()
    height = current_frame.shape[0]
    width = current_frame.shape[1]
    for _h in range(height):
        for _w in range(width):
            if mask[_h, _w] == False:
                current_frame[_h, _w, 0]=255
                current_frame[_h, _w, 1]=255
                current_frame[_h, _w, 2]=255
    
    im_frame = Image.fromarray(current_frame.astype(numpy.uint8))
    return im_frame
