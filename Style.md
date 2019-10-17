# Style Transfer Neural Networks

## Background Research

Here is a link to a bunch of the data used in the network. We could potentially
pull out all of the art based datasets and use those to train our own network.
http://www.robots.ox.ac.uk/~vgg/data/

Here is an example notebook using the pretrained VGG model and tensorflow
https://github.com/ayu34/Neural-Style-Transfer/blob/master/Neural%20Style%20Transfer.ipynb

The actual research paper on VGG
https://arxiv.org/abs/1409.1556

A paper discussing the use of VGG for style-transfer applications
https://arxiv.org/pdf/1508.06576.pdf

Somehow I am not totally understanding how these loss functions are working and
how by running the net you end up with some sort of transformation that just
works.

Basically, the content loss function is calculating the sum of the mean squared
error between the generated image and the content image. The activations in the
higher layers of the generated image often represent objects shown in the
image. So the content loss function focuses on the higher layers correlated
specifically to objects or content.

On the other hand, the style loss function looks at all of the layers in the
CNN. style information is measured as the amount of correlation present between
features maps in a given layer. Next, a loss is defined as the difference of
correlation present between the feature maps computed by the generated image and
the style image.
