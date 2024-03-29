# Halide tutorial lesson 2.

# This lesson demonstrates how to pass in input images.

import os, sys
from halide import *
# The only Halide module  you need is halide. It includes all of Halide

import imageIO

def main():

    # This program defines a single-stage imaging pipeline that
    # brightens an image.

    # First we'll load the input image we wish to brighten.
    # We'll use imageIO to get a numpy array from a PNG image
    im=imageIO.imread('rgb.png')

    # We then create a Halide representation of this image using the Image
    # constructor
    input = Image(Float(32), im)
    # the first input to the Image constructor is a type 32-bit float here)
    # the second can be a filename, a numpy array or nothing
    # when it's a filename, the file gets loaded

    # Next we declare our Func object that represents our one pipeline
    # stage.
    # brighter=Func()
    luminance = Func()

    # Our Func will have three arguments, representing the position
    # in the image and the color channel. Halide treats color
    # channels as an extra dimension of the image, just like in numpy
    # let's declare the corresponding Vars before we can use them
    x, y, c = Var(), Var(), Var()

    # Normally we'd probably write the whole function definition on
    # one line. Here we'll break it apart so we can explain what
    # we're doing at every step.

    # For each pixel of the input image, define an Expr for the input value
    # again note the square brackets
    # value= input[x, y, c]

    # Multiply it by 1.5 to brighten it.
    # value = value * 1.5

    # Finally define the function.
    # brighter[x, y, c] = value

    # The equivalent one-liner to all of the above is:
    #
    # brighter[x, y, c] = input[x, y, c] * 1.5
    #
    luminance[x, y, c] = 0.3 * input[x, y, 0] + 0.6 * input[x, y, 1] + 0.1 * input[x, y, 2]

    # Remember. All we've done so far is build a representation of a
    # Halide program in memory. We haven't actually processed any
    # pixels yet. We haven't even compiled that Halide program yet.

    # So now we'll realize the Func. The size of the output image
    # should match the size of the input image. If we just wanted to
    # brighten a portion of the input image we could request a
    # smaller size. If we request a larger size Halide will throw an
    # error at runtime telling us we're trying to read out of bounds
    # on the input image.
    output = luminance.realize(input.width(), input.height(), input.channels());

    # realize provides us with some Halide internal datatype representing image buffers.
    # We want to convert it to a numpy array. For this, we first turn it into a
    # proper Halide Image using the Halide constructor Image(), and we then convert
    # it to a numpy array. It's a little verbose but not a big deal.
    outputNP=numpy.array(Image(output))

    imageIO.imwrite(outputNP, 'tutorial2.png')

    print "Success!\n"
    return 0;




#the usual Python module business
if __name__ == '__main__':
    main()


# Exercise:
# write Halide code that takes an input RGB image and returns a 1-channel output with its luminance
# the equivalent Python code is below.

def luminance(im):
    output=numpy.empty(im.shape[0], im.shape[1])
    for y in xrange(im.shape[0]):
        for x in xrange(im.shape[1]):
                output[y, x]=0.3*im[y, x, 0]+0.6*im[y, x, 1]+0.1*im[y, x, 2]


