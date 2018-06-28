#!/usr/bin/env python
import argparse

import numpy as np
import theano
from blocks.serialization import load
from theano import tensor
from scipy import misc
#import pickle as p

from keras.datasets import cifar10

from ali import streams


def image_to_z(image):

    """ Perform inference given an image array(s)
    """

    feats = tensor.tensor4('features')

    params = theano.function([feats], 
                             (ali.encoder.mapping.apply(feats)), 
                              allow_input_downcast=True)(image)

    # z is the mean of a gaussian, so take the mean as the best 
    # z encoding for the image
    z_mu = params[:, :ali.encoder._nlat]
    #log_sigma = params[:, ali.encoder._nlat:]

    return z_mu.squeeze()


def z_to_image(z):

    """ Decode z vector to an image array
    """

    decode = ali.decoder.apply(z)
    image = theano.function([], decode)()

    return image[0].transpose(1,2,0).squeeze()

def load_dummy_images(n=2, fn='1.png'):
    # read image as list of numpy images: [(32, 32, 3),...]
    return np.array([misc.imread('1.png') for _ in range(n)])

def preprocess(images):
    # normalize images to 0-1 range
    images = images / 255.0 
    if images.shape[3] == 3:
        # reshapes to channels first format: (n ,3, 32, 32)
        return images.transpose(0,3,1,2)
    else:
        return images
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction using ALI GAN.")
    parser.add_argument("main_loop_path", type=str,
                        help="path to the pickled main loop.")
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print 'CIFAR10 test set shape:', x_test.shape
    x_test = preprocess(x_test)
    print 'CIFAR10 test set fixed shape:', x_test.shape

    #dummy_images = load_dummy_images()
    #print 'Dummy images shape:', dummy_images.shape
    #dummy_images = preprocess(dummy_images)
    #print 'Dummy images fixed shape:', dummy_images.shape

    with open(args.main_loop_path, 'rb') as src:
        main_loop = load(src)
    ali, = main_loop.model.top_bricks
    print('')
    print('ALI Model Loaded...')
    print('')

    zs = image_to_z(x_test)
    print zs.shape

    np.savez_compressed('ali_reps.npz', z_vectors=zs)
    
    #recon = z_to_image(z)
    #misc.imsave("recon.png", recon)


