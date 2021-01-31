import cv2
import numpy
import time

from utils import get_args, read_image, write_image, coloring_seams
from energy import get_energy_A, get_energy_B

all_frames = []


def height_norm(height, max_height):
    if height == -1:
        return 0
    elif height == max_height:
        return height - 1
    else:
        return height


def width_norm(width, max_width):
    if width == -1:
        return 0
    elif width == max_width:
        return width - 1
    else:
        return width


def find_seam(energy, width, height):
    """
        Find minimum seam from energy matrix with specific algorith.
        Returns mask array.
    """
    mask = numpy.full((height, width), True, dtype=bool)
    ancestor = numpy.full((height, width), None)
    cost = numpy.full((height, width), numpy.inf)

    # set cost of first row to their energy
    for w in range(width):
        cost[0, w] = energy[0, w]
    
    # set cost of row from up to down
    # for every pixel find ancestor with minimum cost between up, left_up and right_up ancesor
    for _h in range(1, height):
        for _w in range(width):
            min_cost = float('inf')
            min_ancesor = None

            left_up = (height_norm(_h - 1, height), width_norm(_w - 1, width))
            if cost[left_up] < min_cost:
                min_cost = cost[left_up]
                min_ancesor = left_up

            up = (height_norm(_h - 1, height), width_norm(_w, width))
            if cost[up] < min_cost:
                min_cost = cost[up]
                min_ancesor = up

            right_up = (height_norm(_h - 1, height), width_norm(_w + 1, width))
            if cost[right_up] < min_cost:
                min_cost = cost[right_up]
                min_ancesor = right_up
            
            cost[_h, _w] = cost[min_ancesor] + energy[_h, _w]
            ancestor[_h, _w] = min_ancesor

    # find minimum value of costs in last row
    min_pixel = None
    min_cost = float('inf')
    for _w in range(width):
        if cost[height - 1, _w] < min_cost:
            min_cost = cost[height - 1, _w]
            min_pixel = (height - 1, _w)
    
    # find path from ancestor array and create mask array
    pixel = min_pixel
    for _ in range(height):
        mask[pixel] = False
        pixel = ancestor[pixel]
    
    return mask


def main():
    # get command line arguments
    args = get_args()
    
    # read input image
    image = read_image(args.input_file)

    all_frames.append(image)

    # convert image to array format for do calculations
    im_array = numpy.array(image, dtype=numpy.int16)

    # declare width and height with data of last state of image
    im_width = image.width
    im_height = image.height

    # apply change in width order
    for i in range(args.dx):

        # logging state
        print(f"\nround: {i} -> im_array_width: {im_array.shape[1]}, im_array_height: {im_array.shape[0]}")

        s = time.time()

        # calculate energy function
        energy = get_energy_B(im_array, im_width, im_height)

        print(f"calculate energy array: {time.time() - s} s")
        s = time.time()

        # find seam and create its mask
        mask = find_seam(energy, im_width, im_height)

        print(f"find seam: {time.time() - s} s")

        # make seam white in image and save frame
        frame = coloring_seams(im_array, mask)
        all_frames.append(frame)

        # remove seam from image
        im_array = im_array[mask].reshape(im_height, (im_width - 1), 3)
        im_width -= 1
    

    # rotate image 90 degree for apply seam carving in height direction
    im_array = numpy.rot90(im_array, 1, (1, 0))
    (im_width, im_height) = (im_height, im_width)


    # apply change in height order
    for i in range(args.dy):

        # logging state
        print(f"\nround: {i} -> im_array_width: {im_array.shape[1]}, im_array_height: {im_array.shape[0]}")

        s = time.time()

        # calculate energy function
        energy = get_energy_B(im_array, im_width, im_height)

        print(f"calculate energy array: {time.time() - s} s")
        s = time.time()

        # find seam and create its mask
        mask = find_seam(energy, im_width, im_height)

        print(f"find seam: {time.time() - s} s")

        # make seam white in image and save frame
        frame = coloring_seams(im_array, mask)
        all_frames.append(frame)

        # remove seam from image
        im_array = im_array[mask].reshape(im_height, im_width - 1, 3)
        im_width -= 1
    

    # rotate image to first state
    im_array = numpy.rot90(im_array, -1, (1, 0))

    # create and save gif file
    all_frames[0].save(f"output_of_{args.input_file.split('.')[0]}.gif",
                        save_all=True, append_images=all_frames[1:], duration=500, loop=2)
    
    # create output file
    write_image(im_array, f"output_of_{args.input_file.split('.')[0]}.jpg")



if __name__ == "__main__":
    main()
