from halide import *
import imageIO
import numpy
import a11

def main():
    im=imageIO.imread('rgb.png')
    lumi=im[:,:,1] #I'm lazy, I'll just use green
    smallLumi=numpy.transpose(lumi[0:5, 0:5])

    # Replace if False: by if True: once you have implement the required functions.
    # Exercises:
    if False:
        outputNP, myFunc=a11.smoothGradientNormalized()
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'normalizedGradient.png')
    if False:
        outputNP, myFunc=a11.wavyRGB()
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'rgbWave.png')
    if False:
        outputNP, myFunc=a11.luminance(im)
        imageIO.imwrite(outputNP, 'luminance.png')
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
    if False:
        outputNP, myFunc=a11.sobel(lumi)
        imageIO.imwrite(outputNP, 'sobelMag.png')
        print ' Dimensionality of Halide Func:', myFunc.dimensions()

    if False:
        L=a11.pythonCodeForBoxSchedule5(smallLumi)
        print L
    if False:
        L=a11.pythonCodeForBoxSchedule6(smallLumi)
        print L
    if False:
        L=a11.pythonCodeForBoxSchedule7(smallLumi)
        print L

    if False:
        outputNP, myFunc=a11.localMax(lumi)
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'maxi.png')
    if False:
        input=Image(Float(32), lumi)
        x, y, c = Var('x'), Var('y'), Var('c')
        clamped = Func('clamped')
        clamped[x, y] = input[clamp(x, 0, input.width()-1),
                             clamp(y, 0, input.height()-1)]
        blurX, finalBlur= a11.GaussianSingleChannel(clamped)
    if False:
        im=numpy.load('Input/hk.npy')
        scheduleIndex=0
        outputNP, myFunc=a11.harris(im, scheduleIndex)
        print ' Dimensionality of Halide Func:', myFunc.dimensions()
        imageIO.imwrite(outputNP, 'harris.png')

if __name__ == '__main__':
    main()
