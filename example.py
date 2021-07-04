import numpy as np
from densenet3d import build_densenet
nb_class = 1 
model = build_densenet((121, 145, 121, 1), 0,nb_class)
# Argument: data shape, densenet option, number of class
# input shape: number of subject x 121 x 145 x 121 x 1 -  channel last format
model.summary()
model_weightsf = ''
weight = ''
model.load_weights(model_weightsf)
