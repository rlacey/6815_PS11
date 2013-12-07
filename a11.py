import os, sys
from halide import *
from math import e, pi
import numpy as np

def smoothGradientNormalized():
    '''use Halide to compute a 512x512 smooth gradient equal to x+y divided by 1024
    Do not worry about the schedule.
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    x, y = Var(), Var()
    gradient = Func()

    gradient[x, y] = cast(Float(32), (x + y) / 1024.0)

    output = gradient.realize(512, 512)
    outputNP = numpy.array(Image(output))
    return (outputNP, gradient)


def wavyRGB():
    '''Use a Halide Func to compute a wavy RGB image like that obtained by the following
    Python formula below. output[y, x, c]=(1-c)*cos(x)*cos(y)
    Do not worry about the schedule.
    Hint : you need one more domain dimension than above
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    x, y, c = Var(), Var(), Var()
    wavyRGB = Func()

    wavyRGB[x, y, c] = cast(Float(32), (1 - c) * cos(x) * cos(y))

    output = wavyRGB.realize(400, 400, 3)
    outputNP = numpy.array(Image(output))
    return (outputNP, wavyRGB)


def luminance(im):
    '''input is assumed to be our usual numpy image representation with 3 channels.
    Use Halide to compute a 1-channel image representing 0.3R+0.6G+0.1B
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    input = Image(Float(32), im)
    x, y, c = Var(), Var(), Var()
    luminance = Func()

    luminance[x, y, c] = 0.3 * input[x, y, 0] + 0.6 * input[x, y, 1] + 0.1 * input[x, y, 2]

    output = luminance.realize(input.width(), input.height(), input.channels())
    outputNP = numpy.array(Image(output))
    return (outputNP, luminance)


def  sobel(lumi):
    ''' lumi is assumed to be a 1-channel numpy array.
    Use Halide to apply a Sobel filter and return the gradient magnitude.
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    input = Image(Float(32), lumi)
    x, y = Var(), Var()
    sx, sy, sobelMagnitude, clamped = Func(), Func(), Func(), Func()

    clamped[x, y] = input[clamp(x, 0, input.width() - 1),
                          clamp(y, 0, input.height() - 1)]

    sx[x,y] = 0.25 * (
                -1 * clamped[x-1, y-1] + 1 * clamped[x-1, y+1]
                -2 * clamped[x  , y-1] + 2 * clamped[x  , y+1]
                -1 * clamped[x+1, y-1] + 1 * clamped[x+1, y+1])

    sy[x,y] = 0.25 * (
                -1 * clamped[x-1, y-1] + 1 * clamped[x+1, y-1]
                -2 * clamped[x-1, y  ] + 2 * clamped[x+1, y  ]
                -1 * clamped[x-1, y+1] + 1 * clamped[x+1, y+1])

    sobelMagnitude[x,y] = sqrt(sx[x, y] ** 2 + sy[x, y] ** 2)

    output = sobelMagnitude.realize(input.width(), input.height())
    outputNP=numpy.array(Image(output))
    return (outputNP, sobelMagnitude)


def pythonCodeForBoxSchedule5(lumi):
    ''' lumi is assumed to be a 1-channel numpy array.
    Write the python nested loops corresponding to the 3x3 box schedule 5
    and return a list representing the order of evaluation.
    Each time you perform a computation of blur_x or blur_y, put a triplet with the name
    of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y.
    e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''

    # schedule 5:
    # blur_y.compute_root()
    # blur_x.compute_at(blur_y, x)

    schedule = []
    input = Image(Float(32), lumi)
    width, height = input.width()-2, input.height()-2
    for y in xrange(height):
        for x in xrange(width):
            schedule.append(("blur_x", x,     y))
            schedule.append(("blur_x", x + 1, y))
            schedule.append(("blur_x", x + 2, y))
            schedule.append(("blur_y", x,     y))
    return schedule


def pythonCodeForBoxSchedule6(lumi):
    ''' lumi is assumed to be a 1-channel numpy array.
    Write the python nested loops corresponding to the 3x3 box schedule 5
    and return a locationst representing the order of evaluation.
    Each time you perform a computation of blur_x or blur_y, put a triplet with the name
    of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y.
    e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''

    # schedule 6:
    # blur_y.tile(x, y, xo, yo, xi, yi, 2, 2)
    # blur_x.compute_at(blur_y, yo)

    schedule = []
    (height, width) = (np.shape(lumi)[0], np.shape(lumi)[1])
    for yo in xrange((height + 1) / 2):
        # BLUR_X
        for yi in xrange(4):
            y = 2 * yo + yi
            y = np.minimum(y, height + 1)
            for xi in xrange(width):
                schedule.append(("blur_x", xi, y))
        for xo in xrange((width + 1) / 2):
            # BLUR_Y
            for yi in xrange(2):
                y = 2 * yo + yi
                y = np.minimum(y, height - 1)
                for xi in xrange(2):
                    x = 2 * xo + xi
                    x = np.minimum(x, width - 1)
                    schedule.append(("blur_y", x, y))
    return schedule


def pythonCodeForBoxSchedule7(lumi):
    ''' lumi is assumed to be a 1-channel numpy array.
    Write the python nested loops corresponding to the 3x3 box schedule 5
    and return a list representing the order of evaluation.
    Each time you perform a computation of blur_x or blur_y, put a triplet with the name
    of the function (string 'blur_x' or 'blur_y') and the output coordinates x and y.
    e.g. [('blur_x', 0, 0), ('blur_y', 0,0), ('blur_x', 0, 1), ...] '''

    # schedule 7
    # blur_y.split(x, xo, xi, 2)
    # blur_x.compute_at(blur_y, y)

    schedule = []
    (height, width) = (np.shape(lumi)[0]-2, np.shape(lumi)[1]-2)
    for yo in xrange(height):
        # COMPUTE BLUR_X
        for yi in xrange(3):
            for xi in xrange(width):
                schedule.append(("blur_x", xi, yo + yi))
        for xo in xrange(width / 2):
            for xi in xrange(2):
                schedule.append(("blur_y", xo + 2 * xi, yo))
    return schedule


########### PART 2 ##################

def localMax(lumi):
    ''' the input is assumed to be a 1-channel image
    for each pixel, return 1.0 if it's a local maximum and 0.0 otherwise
    Don't forget to handle pixels at the boundary.
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    input = Image(Float(32), lumi)
    x, y = Var(), Var()
    thresholding, clamped = Func(), Func()

    clamped[x, y] = input[clamp(x, 0, input.width() - 1),
                             clamp(y, 0, input.height() - 1)]

    thresholding[x, y] = select((input[x, y] > clamped[x-1, y  ]) &
                                (input[x, y] > clamped[x+1, y  ]) &
                                (input[x, y] > clamped[x  , y-1]) &
                                (input[x, y] > clamped[x  , y+1])
                                , 1.0, 0.0)

    output = thresholding.realize(input.width(), input.height())
    outputNP=numpy.array(Image(output))
    return (outputNP, thresholding)


def GaussianSingleChannel(input, sigma = 5, trunc=3):
    '''takes a single-channel image or Func IN HALIDE FORMAT as input
        and returns a Gaussian blurred Func with standard
        deviation sigma, truncated at trunc*sigma on both sides
        return two Funcs corresponding to the two stages blurX, blurY. This will be
        useful later for scheduling.
        We advise you use the sum() sugar
        We also advise that you first generate the kernel as a Halide Func
        You can assume that input is a clamped image and you don't need to worry about
        boundary conditions here. See calling example in test file. '''

    bound = int(sigma * trunc)
    kernel_width = int(2 * bound + 1)
    x, y, c, weight_sum = Var(), Var(), Var(), Var()
    kernel, gaussian, blur_x, blur_y = Func(), Func(), Func(), Func()
    r = RDom(-bound, kernel_width)

    kernel[x,c] = e ** (-1 * (x - c)**2 / (2 * float(sigma)**2))
    weight_sum = sum(kernel[r.x, 0])
    gaussian[x,c] = kernel[x, c] / weight_sum

    blur_x[x,y] = sum(gaussian[x+r.x, x] * input[x+r.x, y])
    blur_y[x,y] = sum(gaussian[y, y+r.x] * blur_x[x, y+r.x])

    gaussian.compute_root()
    blur_x.compute_root()

    return (blur_x, blur_y)


def harris(im, scheduleIndex, tileX1 = 128, tileY1 = 128, tileX2 = 128, tileY2 = 128, tileX3 = 128, tileY3 = 128, vector = 4):
    ''' im is a numpy RGB array.
    return the location of Harris corners like the reference Python code, but computed
    using Halide.
    when scheduleIndex is zero, just schedule all the producers of non-local consumers as root.
    when scheduleIndex is 1, use a smart schedule that makes use of parallelism and
    has decent locality (tiles are often a good option). Do not worry about vectorization.
    Note that the local maximum criterion is simplified compared to our original Harris
    You might want to reuse or copy-paste some of the code you wrote above
    Return a pair (outputNP, myFunc) where outputNP is a numpy array and myFunc is a Halide Func'''

    # CONSTANTS
    sigma = 5
    trunc = 3

    # COMMON HALIDE VARS
    x, y = Var(), Var()
    input = Image(Float(32), im)

    # COMPUTE LUMINANCE
    c = Var()
    luminance, lumi = Func(), Func()
    luminance[x, y, c] = 0.3 * input[x, y, 0] + 0.6 * input[x, y, 1] + 0.1 * input[x, y, 2]
    lumi[x, y] = luminance[x, y, 1]

    # PREVENT BOUNDARY ISSUES
    clamped = Func()
    clamped[x, y] = lumi[clamp(x, 0, input.width()-1),
                         clamp(y, 0, input.height()-1)]

    # APPLY GAUSSIAN BLUR TO LUMINANCE
    bound = int(sigma * trunc)
    kernel_width = int(2 * bound + 1)

    weight_sum = Var()
    kernel, gaussian, blur_x, blur_y = Func(), Func(), Func(), Func()
    r = RDom(-bound, kernel_width)

    kernel[c] = e ** (-1 * (c)**2 / (2 * float(sigma)**2))
    weight_sum = sum(kernel[r.x])
    gaussian[c] = kernel[c] / weight_sum

    blur_x[x,y] = sum(gaussian[r.x] * clamped[x+r.x, y])
    blur_y[x,y] = sum(gaussian[r.x] * blur_x[x, y+r.x])

    # COMPUTE GRADIENT MAGNITUDE
    sx, sy, sobelMagnitude, clamped = Func(), Func(), Func(), Func()
    clamped[x, y] = blur_y[clamp(x, 0, input.width() - 1),
                           clamp(y, 0, input.height() - 1)]
    sx[x,y] = 0.25 * (
                -1 * blur_y[x-1, y-1] + 1 * blur_y[x-1, y+1]
                -2 * blur_y[x  , y-1] + 2 * blur_y[x  , y+1]
                -1 * blur_y[x+1, y-1] + 1 * blur_y[x+1, y+1])
    sy[x,y] = 0.25 * (
                -1 * blur_y[x-1, y-1] + 1 * blur_y[x+1, y-1]
                -2 * blur_y[x-1, y  ] + 2 * blur_y[x+1, y  ]
                -1 * blur_y[x-1, y+1] + 1 * blur_y[x+1, y+1])

    # FORM TENSOR
    ix2, iy2, ixiy = Func(), Func(), Func()
    ix2[x,y] = (sx[x, y])**2
    iy2[x,y] = (sy[x, y])**2
    ixiy[x,y] = (sx[x, y] * sy[x, y])

    # BLUR TENSOR
    ix2_x, iy2_x, ixiy_x, ix2_blur, iy2_blur, ixiy_blur = Func(), Func(), Func(), Func(), Func(), Func()

    ix2_x[x,y] = sum(gaussian[r.x] * ix2[x+r.x, y])
    ix2_blur[x,y] = sum(gaussian[r.x] * ix2_x[x, y+r.x])

    iy2_x[x,y] = sum(gaussian[r.x] * iy2[x+r.x, y])
    iy2_blur[x,y] = sum(gaussian[r.x] * iy2_x[x, y+r.x])

    ixiy_x[x,y] = sum(gaussian[r.x] * ixiy[x+r.x, y])
    ixiy_blur[x,y] = sum(gaussian[r.x] * ixiy_x[x, y+r.x])

    # DETERMINANT OF TENSOR
    det = Func()
    det[x,y] = ix2_blur[x,y] * iy2_blur[x,y] - ixiy_blur[x,y]**2

    # TRACE OF TENSOR
    trace = Func()
    trace[x,y] = ix2_blur[x,y] + iy2_blur[x,y]

    # HARRIS RESPONSE
    M = Func()
    M[x,y] = det[x,y] - 0.15 * trace[x,y]**2

    # THRESHOLD
    threshold = Func()
    threshold[x,y] = select(M[x,y] > 0.0, 1.0, 0.0)

    # LOCAL MAXIMUM
    thresholded =Func()
    thresholded[x,y] = select((M[x, y] > M[x-1, y  ]) &
                              (M[x, y] > M[x+1, y  ]) &
                              (M[x, y] > M[x  , y-1]) &
                              (M[x, y] > M[x  , y+1]) &
                              (threshold[x,y] > 0)
                              , 1.0, 0.0)


    gaussian.compute_root()
    blur_x.compute_root()
    blur_y.compute_root()
    sx.compute_root()
    sy.compute_root()
    ix2.compute_root()
    iy2.compute_root()
    ixiy.compute_root()
    ix2_x.compute_root()
    ix2_blur.compute_root()
    iy2_x.compute_root()
    iy2_blur.compute_root()
    ixiy_x.compute_root()
    ixiy_blur.compute_root()

    if scheduleIndex == 1:

        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')

        thresholded.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)

        blur_y.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        blur_x.compute_at(blur_y, xo)

        ix2_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        ix2_x.compute_at(ix2_blur, xo)

        iy2_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        iy2_x.compute_at(iy2_blur, xo)

        ixiy_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        ixiy_x.compute_at(ixiy_blur, xo)

    elif scheduleIndex == 2:

        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')

        thresholded.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)

        blur_y.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        blur_x.compute_at(blur_y, yo)

        ix2_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        ix2_x.compute_at(ix2_blur, yo)

        iy2_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        iy2_x.compute_at(iy2_blur, yo)

        ixiy_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        ixiy_x.compute_at(ixiy_blur, yo)

    elif scheduleIndex == 3:

        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')

        thresholded.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)

        blur_y.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        blur_x.compute_at(blur_y, xo)

        ix2_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        ix2_x.compute_at(ix2_blur, xo)

        iy2_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        iy2_x.compute_at(iy2_blur, xo)

        ixiy_blur.tile(x, y, xo, yo, xi, yi, tileX, tileY).parallel(yo)
        ixiy_x.compute_at(ixiy_blur, xo)

    elif scheduleIndex == 4:

        xi, yi, xo, yo=Var('xi'), Var('yi'), Var('xo'), Var('yo')

        thresholded.tile(x, y, xo, yo, xi, yi, tileX1, tileY1).parallel(yo)

        blur_y.tile(x, y, xo, yo, xi, yi, tileX2, tileY2).parallel(yo).vectorize(xi, vector)
        blur_x.compute_at(blur_y, xo).vectorize(x, 4)

        ix2_blur.tile(x, y, xo, yo, xi, yi, tileX3, tileY3).parallel(yo).vectorize(xi, vector)
        ix2_x.compute_at(ix2_blur, xo).vectorize(x, 8)

        iy2_blur.tile(x, y, xo, yo, xi, yi, tileX3, tileY3).parallel(yo).vectorize(xi, vector)
        iy2_x.compute_at(iy2_blur, xo).vectorize(x, 8)

        ixiy_blur.tile(x, y, xo, yo, xi, yi, tileX3, tileY3).parallel(yo).vectorize(xi, vector)
        ixiy_x.compute_at(ixiy_blur, xo).vectorize(x, 8)

    else:
        pass

    output = thresholded.realize(input.width(), input.height())
    outputNP=numpy.array(Image(output))
    return outputNP, thresholded


def runAndMeasure(myFunc, w, h, nTimes=5):
    L=[]
    output=None
    # print 'Compiling...'
    myFunc.compile_jit()
    for i in xrange(nTimes):
        # print 'iter:', i
        t=time.time()
        output = myFunc.realize(w,h)
        L.append (time.time()-t)
    # print 'running times :', L
    # print 'done'
    hIm=Image(output)
    mpix=hIm.width()*hIm.height()/1e6
    # print 'best: ', numpy.min(L), 'average: ', numpy.mean(L)
    # print  '%.5f ms per megapixel (%.7f ms for %d megapixels)' % (numpy.mean(L)/mpix*1e3, numpy.mean(L)*1e3, mpix)
    return numpy.mean(L)/mpix*1e3

# def initial_optimizer(im, w, h):
#     schedules = range(5)
#     sizes = [32, 64, 128, 256]
#     # time, schedule, x, y
#     minimum = (float('inf'), None, None, None)
#     for i in schedules:
#         for tile_x in sizes:
#             for tile_y in sizes:
#                 func = harris(im, i, tile_x, tile_y)[1]
#                 time = runAndMeasure(func, w, h, 5)
#                 print 'schedule', i, ' (', tile_x, tile_y, ') :', time
#                 if time < minimum[0]:
#                     minimum = (time, i, tile_x, tile_y)
#                     print 'Updated minimum', minimum
#     print minimum
#     return minimum

def optimizer(im, w, h):
    sizes = [32, 64, 128, 256]
    vectors = [2, 4, 8]
    # time, schedule, x, y
    minimum = (float('inf'), None, None, None, None, None, None, None)
    for tile_x_1 in sizes:
        for tile_y_1 in sizes:
            for tile_x_2 in sizes:
                for tile_y_2 in sizes:
                    for tile_x_3 in sizes:
                        for tile_y_3 in sizes:
                            for vec in vectors:
                                func = harris(im, 4, tile_x_1, tile_y_1, tile_x_2, tile_y_2, tile_x_3, tile_y_3, vec)[1]
                                time = runAndMeasure(func, w, h, 5)
                                print '(', tile_x_1, tile_y_1, tile_x_2, tile_y_2, tile_x_3, tile_y_3, ') V', vec, ' : ', time
                                if time < minimum[0]:
                                    minimum = (time, tile_x_1, tile_y_1, tile_x_2, tile_y_2, tile_x_3, tile_y_3, vec)
                                    print 'Updated minimum', minimum
    print 'Best', minimum
    return minimum

# im = numpy.load('Input/rgb.npy')
# runAndMeasure(harris(im, 0)[1], 7319, 4885, nTimes=5)
# optimizer(im, 1536, 2560)




