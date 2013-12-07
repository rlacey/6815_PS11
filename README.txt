Name: Ryan Lacey
MIT Email: rlacey@mit.edu

Q1:How long did the assignment take?:
{A1:
    18 hours
}

Q2:Potential issues with your solution and explanation of partial completion (for partial credit):
{A2:
    None
}

Q3:Anything extra you may have implemented:
{A3:
    None
}

Q4:Collaboration acknowledgement (but again, you must write your own code):
{A4:
    cliu2014,
    idoe
}

Q5:What was most unclear/difficult?:
{A5:
    Translating halide schedules into python. The different scheduling options are difficult to understand.
}

Q6:What was most exciting?:
{A6:
    Making functions we've implemented before was sorta cool, but this pset wasn't nearly as exciting as previous ones because we were reimplementing old stuff.
}

Q7: How long did it take for the 2 schedules for the smooth gradient?
{A7:
    default_gradient: 0.00140309333801s,
    fast_gradient:    0.00310397148132s
}

Q8: Speed in ms per megapixel for the 4 schedules (1 per line)
{A8:
    ROOT:   6.92177 ms per megapixel (247.3071575 ms for 35 megapixels)
    INLINE: 5.24960 ms per megapixel (187.5624180 ms for 35 megapixels)
    TILING: 5.80219 ms per megapixel (207.3058605 ms for 35 megapixels)
    T & P:  2.32685 ms per megapixel (83.1359863 ms for 35 megapixels)
}

Q9: What machine did you use (CPU type, speed, number of cores, memory)
{A9:
    1.3 GHz Intel Core i5,
    dual-core,
    8GB 1600 MHz DDR3
}

Q10: Speed for the box schedules, and best tile size
{A10:
    Schedule5: 0.000371932983398s,
    Schedule6: 0.00040602684021s,
    Schedule7: 0.00004506111145s
}

Q11: How fast did Fredo’s Harris and your two schedules were on your machine?
{A11:
    Couldn't run Fredo's code :\ SciPy would not accept it as given
    COMPUTE ROOT:      342.03170 ms per megapixel (12228.7671566 ms for 35 megapixels)
    Tiling & Paralell: 105.70527 ms per megapixel (3779.3138027 ms for 35 megapixels)
}

Q12: Describe your auto tuner in one paragraph
{A12:
    The autotuner goes through several permutations of tile sizes. There are three groups of tiles: luminance blur, tensor blur, and the final thresholding function. Additionally the parallelization is vectorized and each tile combinations is paired with the same vectorization constant. The tile dimensions and vector size that result in the fastest runtime are chosen as the optimal configuration.
}

Q13: What is the best schedule you found for Harris.
{A13:
    time (ms/MP, threshold tile X, threshold tile Y, luminance blur tile X, luminance blur tile Y, tensor blur tile X, tensor blur tile Y, vector)
    (98.826531029772013, 32, 64, 64, 256, 32, 128, 8)
}