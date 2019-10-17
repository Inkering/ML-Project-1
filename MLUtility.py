# Utilities for Neural Style Transfer
# Written by David Tarazi and Dieter Brehm
# October 2019
#
# See accompanying Jupyter notebook for implementation
# and analysis of style transfer using neural networks

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import time
import functools

def transform_image(img_path, dim):
    """ image scaling and reading from a file path"""
    # load the image into a variable
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # look at size of the image, get the longest dimension
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)

    # calculate a scale factor to fit to the desired dims
    scale = dim / long_dim

    # calculate the new dimensions
    new_shape = tf.cast(shape * scale, tf.int32)

    # resize the image vector
    img = tf.image.resize(img, new_shape)

    # what does this do?
    img = img[tf.newaxis, :]

    return img


def imshow(image, title=None):
    """ quick function for plotting an image, gets rid of num_frames """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def vgg_2_functional_model(layer_names):
    # download the model
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # output vectors (tensors) for each layer we care about
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # input vectors (tensors) as the input layer (first layer) of the model
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    """
    Finds the correlation matrix for style loss and sums
    all of the feature correlations for a particular layer
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)

    # num_locations is the same as number of pixels
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    # normalize
    return result/(num_locations)

def covar_matrix(input_tensor):
    """
    Finds the covariance matrix for style loss and sums
    all of the feature correlations for a particular layer

    The difference between this and gram_matrix() is the order
    of transpose (GG' for gram vs G'G for covar). We use this
    to test which type of correlation matrix is better
    """
    result = tfp.stats.covariance(input_tensor)
    input_shape = tf.shape(input_tensor)

    # num_locations is the same as number of pixels
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    # normalize
    return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
    """
    Creates a model class that has all the properties of a keras Model with specific style and content layers.
    When called, it takes the inputs and returns a style and content dictionary with the layer and its output.
    """
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_2_functional_model(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg_trainable = False

    def call(self, inputs, gram = True):
        """
        Overloading the keras Model call method

        Input False for gram to use the covariance matrix
        """
        # Expects float input in [0,1]
        inputs = inputs*255.0
        # depending on model - resize/process inputs to the right size, shape, etc.
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        # image layers
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # Puts the style outputs into their correlation matrix form so that we can compute loss
        if gram is True:
            style_outputs = [gram_matrix(style_output)
                             for style_output in style_outputs]
        else:
            style_outputs = [covar_matrix(style_output)
                             for style_output in style_outputs]

        content_dict = {content_name:value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


class experiment_handler():
    def __init__(self,
                 content_image,
                 model_instance,
                 style_targets,
                 content_targets,
                 style_weight,
                 content_weight,
                 num_style_layers,
                 num_content_layers):
        self.content_image = content_image
        self.model_instance = model_instance
        self.style_targets = style_targets
        self.content_targets = content_targets
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.num_style_layers = num_style_layers
        self.num_content_layers = num_content_layers


    def style_content_loss(self, outputs):
        """
        Finding the loss for both style and content by comparing the generated image to the input image
        """
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss


    def train_step(self, image):
        #Tape keeps track of the gradient at every step and differentiates between the weights at each node
        with tf.GradientTape() as tape:
            outputs = self.model_instance(image)
            loss = self.style_content_loss(outputs)
            self.losses.append(loss)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        self.image.assign(clip_0_1(image))


    def run(self, epoch):
        self.opt = tf.train.AdamOptimizer(learning_rate=0.05, beta1=0.99, epsilon=1e-1)
        self.image = tf.Variable(self.content_image)
        self.losses = []

        for i in range(epoch):
            self.train_step(self.image)
            if (i % 100) == 0:
                plt.figure()
                plt.imshow(tensor_to_image(self.image))
