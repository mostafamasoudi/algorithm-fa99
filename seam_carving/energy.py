import cv2
import numpy

def get_energy_B(im_array, width, height):
    height = im_array.shape[0]
    width = im_array.shape[1]
    im = cv2.cvtColor(im_array.astype(numpy.uint8), cv2.COLOR_BGR2GRAY).astype(numpy.float64)

    energy = numpy.zeros((height, width))
    m = numpy.zeros((height, width))
    
    U = numpy.roll(im, 1, axis=0)
    L = numpy.roll(im, 1, axis=1)
    R = numpy.roll(im, -1, axis=1)
    
    cU = numpy.abs(R - L)
    cL = numpy.abs(U - L) + cU
    cR = numpy.abs(U - R) + cU
    
    for i in range(1, height):
        mU = m[i-1]
        mL = numpy.roll(mU, 1)
        mR = numpy.roll(mU, -1)
        
        mULR = numpy.array([mU, mL, mR])
        cULR = numpy.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = numpy.argmin(mULR, axis=0)
        m[i] = numpy.choose(argmins, mULR)
        energy[i] = numpy.choose(argmins, cULR)   
        
    return energy


def get_energy_A(im_array, width, height):
    ''' 
        Calculate energy array according to rgb value of pixels 
        O(n ^ 2)
    '''
    resized_arr = numpy.pad(im_array, ((1, 1), (1, 1), (0, 0)), 'edge')

    energy = numpy.zeros((height, width), dtype=int)
    for _h in range(1, height+1):
        for _w in range(1, width+1):
            hor_right = resized_arr[_h, _w + 1]
            hor_left = resized_arr[_h, _w - 1]
            ex = ((hor_right[0] - hor_left[0]) ** 2
                + (hor_right[1] - hor_left[1]) ** 2
                + (hor_right[2] - hor_left[2]) ** 2
                )

            ver_up = resized_arr[_h - 1 , _w]
            ver_down = resized_arr[_h + 1, _w]
            ey = ((ver_down[0] - ver_up[0]) ** 2
                + (ver_down[1] - ver_up[1]) ** 2
                + (ver_down[2] - ver_up[2]) ** 2)
            
            energy[_h - 1, _w - 1] = numpy.sqrt(ex + ey)

    return energy