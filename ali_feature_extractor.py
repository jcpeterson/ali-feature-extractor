#!/usr/bin/env python
import argparse

import numpy
import theano
from blocks.serialization import load
from matplotlib import cm, pyplot
from mpl_toolkits.axes_grid1 import ImageGrid
from theano import tensor
from scipy import misc
from os import listdir
from os.path import isfile, join
import pickle as p

from ali import streams




def image_to_z(main_loop, image):

    """ Perform inference given an image filename. 
    """

    ali, = main_loop.model.top_bricks

    x = tensor.tensor4('features')

    params = theano.function([x], (ali.encoder.mapping.apply(x)), allow_input_downcast=True)(image)
    mu, log_sigma = params[:, :ali.encoder._nlat], params[:, ali.encoder._nlat:]
    return mu



def z_to_image(main_loop,z_mu_vector):
    ali, = main_loop.model.top_bricks
    decode = ali.decoder.apply(z_mu_vector)
    finalImage = theano.function([], decode)()
    return finalImage[0].transpose(1,2,0).squeeze()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction using ALI GAN.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    args = parser.parse_args()


    image = [misc.imread('1.png')]
    print image.shape
    image = (numpy.array(image)/255.0).transpose(0,3,1,2)
    print image.shape
    exit()

    with open(args.main_loop_path, 'rb') as src:
        main_loop = load(src)

    z_vector = image_to_z(main_loop, args.image_name)

    #finalImage = z_to_image(main_loop, z_vector)
    #misc.imsave("6r.png",finalImage)


