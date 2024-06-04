import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint
import numpy as np
from tensorflow.keras import backend as K


from tensorflow.keras.layers import Input, Lambda, Layer, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, Add, Subtract, Cropping2D, Reshape, Dense
from tensorflow.keras.models import Model  # C'est ici qu'il faut importer Model
from tensorflow.keras import initializers, constraints

class Localization(tf.keras.layers.Layer):
    def __init__(self):
        super(Localization, self).__init__()
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv1 = tf.keras.layers.Conv2D(20, [5, 5], activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(20, [5, 5], activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(20, activation='relu')
        self.fc2 = tf.keras.layers.Dense(6, activation=None, bias_initializer=tf.keras.initializers.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), kernel_initializer='zeros')

    def build(self, input_shape):
        print("Building Localization Network with input shape:", input_shape)

    def compute_output_shape(self, input_shape):
        return [None, 6]

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        theta = self.fc2(x)
        theta = tf.keras.layers.Reshape((2, 3))(theta)
        return theta

def sample(img, coords):
    """
    Args:
        img: bxhxwxc
        coords: bxh2xw2x2. each coordinate is (y, x) integer.
            Out of boundary coordinates will be clipped.
    Return:
        bxh2xw2xc image
    """
    shape = img.get_shape().as_list()[1:]   # h, w, c
    batch = tf.shape(img)[0]
    shape2 = coords.get_shape().as_list()[1:3]  # h2, w2
    assert None not in shape2, coords.get_shape()
    max_coor = tf.constant([shape[0] - 1, shape[1] - 1], dtype=tf.float32)

    coords = tf.clip_by_value(coords, 0., max_coor)  # borderMode==repeat
    coords = tf.cast(coords, tf.int32)

    batch_index = tf.range(batch, dtype=tf.int32)
    batch_index = tf.reshape(batch_index, [-1, 1, 1, 1])
    batch_index = tf.tile(batch_index, [1, shape2[0], shape2[1], 1])    # bxh2xw2x1
    indices = tf.concat([batch_index, coords], axis=3)  # bxh2xw2x3
    sampled = tf.gather_nd(img, indices)
    return sampled
    
def GridSample(inputs, borderMode='repeat'):
    """
    Sample the images using the given coordinates, by bilinear interpolation.
    This was described in the paper:
    `Spatial Transformer Networks <http://arxiv.org/abs/1506.02025>`_.
    This is equivalent to `torch.nn.functional.grid_sample`,
    up to some non-trivial coordinate transformation.
    This implementation returns pixel value at pixel (1, 1) for a floating point coordinate (1.0, 1.0).
    Note that this may not be what you need.
    Args:
        inputs (list): [images, coords]. images has shape NHWC.
            coords has shape (N, H', W', 2), where each pair of the last dimension is a (y, x) real-value
            coordinate.
        borderMode: either "repeat" or "constant" (zero-filled)
    Returns:
        tf.Tensor: a tensor named ``output`` of shape (N, H', W', C).
    """
    image, mapping = inputs
    assert image.get_shape().ndims == 4 and mapping.get_shape().ndims == 4
    input_shape = image.get_shape().as_list()[1:]
    assert None not in input_shape, \
        "Images in GridSample layer must have fully-defined shape"
    assert borderMode in ['repeat', 'constant']

    orig_mapping = mapping
    mapping = tf.maximum(mapping, 0.0)
    lcoor = tf.floor(mapping)
    ucoor = lcoor + 1

    diff = mapping - lcoor
    neg_diff = 1.0 - diff  # bxh2xw2x2

    lcoory, lcoorx = tf.split(lcoor, 2, 3)
    ucoory, ucoorx = tf.split(ucoor, 2, 3)

    lyux = tf.concat([lcoory, ucoorx], 3)
    uylx = tf.concat([ucoory, lcoorx], 3)

    diffy, diffx = tf.split(diff, 2, 3)
    neg_diffy, neg_diffx = tf.split(neg_diff, 2, 3)

    ret = tf.add_n([sample(image, lcoor) * neg_diffx * neg_diffy,
                    sample(image, ucoor) * diffx * diffy,
                    sample(image, lyux) * neg_diffy * diffx,
                    sample(image, uylx) * diffy * neg_diffx], name='sampled')
    if borderMode == 'constant':
        max_coor = tf.constant([input_shape[0] - 1, input_shape[1] - 1], dtype=tf.float32)
        mask = tf.greater_equal(orig_mapping, 0.0)
        mask2 = tf.less_equal(orig_mapping, max_coor)
        mask = tf.logical_and(mask, mask2)  # bxh2xw2x2
        mask = tf.reduce_all(mask, [3])  # bxh2xw2 boolean
        mask = tf.expand_dims(mask, 3)
        ret = ret * tf.cast(mask, tf.float32)
    return tf.identity(ret, name='output')

class BilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, height=40, width=40):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }
    
    def build(self, input_shape):
        print("Building Bilinear Interpolation Layer with input shape:", input_shape)

    def advance_indexing(self, inputs, x, y):
        '''
        Numpy like advance indexing is not supported in tensorflow, hence, this function is a hack around the same method
        '''        
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]

        x = tf.clip_by_value(x, 0, self.width-1)
        y = tf.clip_by_value(y, 0, self.height-1)
        
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        #print(theta)
        return self.interpolate(images, homogenous_coordinates, theta)

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)
            
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates
    
    def interpolate(self, images, homogenous_coordinates, theta):

        xys = np.array([(y, x, 1) for y in range(self.width)
                        for x in range(self.height)], dtype='float32')
        xys = tf.constant(xys, dtype=tf.float32, name='xys')    # p x 3
        stn = tf.reshape(theta, [-1, 2, 3], name='affine')  # bx2x3
        stn = tf.reshape(tf.transpose(stn, [2, 0, 1]), [3, -1])  # 3 x (bx2)
        coor = tf.reshape(tf.matmul(xys, stn),
                          [self.width, self.height, -1, 2])
        coor = tf.transpose(coor, [2, 0, 1, 3], 'sampled_coords')  # b h w 2
        sampled = GridSample( [images, coor], borderMode='constant')

        return sampled


    def __init__(self, height=40, width=40):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }
    
    def build(self, input_shape):
        print("Building Bilinear Interpolation Layer with input shape:", input_shape)

    def advance_indexing(self, inputs, x, y):
        '''
        Numpy like advance indexing is not supported in tensorflow, hence, this function is a hack around the same method
        '''        
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]
        
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        return self.interpolate(images, homogenous_coordinates, theta)

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)
            
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates
    
    def interpolate(self, images, homogenous_coordinates, theta):

        with tf.name_scope("Transformation"):
            transformed = tf.matmul(theta, homogenous_coordinates)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(transformed, [-1, self.height, self.width, 2])
            
            #print(transformed)
            #print(homogenous_coordinates)
            
                
            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]
                
            x = ((x_transformed + 1.) * tf.cast(self.width, dtype=tf.float32)) * 0.5
            y = ((y_transformed + 1.) * tf.cast(self.height, dtype=tf.float32)) * 0.5
            
        
        with tf.name_scope("VariableCasting"):
            x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, self.width-1)
            x1 = tf.clip_by_value(x1, 0, self.width-1)
            y0 = tf.clip_by_value(y0, 0, self.height-1)
            y1 = tf.clip_by_value(y1, 0, self.height-1)
            x = tf.clip_by_value(x, 0, tf.cast(self.width, dtype=tf.float32)-1.0)
            y = tf.clip_by_value(y, 0, tf.cast(self.height, dtype=tf.float32)-1)

        with tf.name_scope("AdvanceIndexing"):
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"):
            x0 = tf.cast(x0, dtype=tf.float32)
            x1 = tf.cast(x1, dtype=tf.float32)
            y0 = tf.cast(y0, dtype=tf.float32)
            y1 = tf.cast(y1, dtype=tf.float32)
                            
            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)

            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)
                           
        return tf.math.add_n([wa*Ia + wb*Ib + wc*Ic + wd*Id])

# This function creates a custom layer in the neural network 
class CustomConv( Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dims = output_dims

        super(CustomConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.output_dims,
                                      initializer='ones',
                                      trainable=True)

        super(CustomConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x,imSize):
        a = tf.keras.backend.reshape(x, shape=(-1,imSize,imSize,1))
        return tf.multiply(a,self.kernel)

    def compute_output_shape(self, input_shape):
        return (self.output_dims)

# This function creates a custom layer in the neural network 
# which takes only the led position that is being used in the training example 
class take_one(Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dims = output_dims

        super(take_one, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.output_dims,
                                      initializer='ones',
                                      trainable=False)

        super(take_one, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        b = tf.keras.backend.reshape(x[:,0], shape=(1,1))
        a = tf.keras.backend.cast(b, dtype='float32')
        return a[:,0]*1

    def compute_output_shape(self, input_shape):
        return (self.output_dims)

# This function limits the weight value between a minimum and maximum value 
class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value =  min_value
        self.max_value = max_value

    def __call__(self, w):        
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}


# This function is a custom neural network layer that does the linear matching between inputs and weights
class ConvexCombination(Layer):
    def __init__(self, **kwargs):
        super(ConvexCombination, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lambd2 = self.add_weight(name='lambda2',
                                     shape=(10,1),  # Adding one dimension for broadcasting
                                     initializer='ones',  # Try also 'ones' and 'uniform'
                                     trainable=True,
                                     constraint =Between(-1.,1.))
        super(ConvexCombination, self).build(input_shape)

    def call(self, x):
        # x is a list of two tensors with shape=(batch_size, H, T)
        h1,h2,h3,h4,h5,h6,h7,h8,h9,h10 = x
        a= self.lambd2[0,0]
        b= self.lambd2[1,0]
        c= self.lambd2[2,0]
        d= self.lambd2[3,0]
        e= self.lambd2[4,0]
        f= self.lambd2[5,0]
        g= self.lambd2[6,0]
        h= self.lambd2[7,0]
        i= self.lambd2[8,0]
        j= self.lambd2[9,0]
        # k= self.lambd2[10,0]
        # l= self.lambd2[11,0]

        new_ctf = a*h1 + b*h2 +  c*h3 + d*h4 + e*h5 + f*h6 + g*h7 + h*h8+i*h9 +j*h10#+ k*h11 +l*h12
        return new_ctf

    def compute_output_shape(self, input_shape):
        return input_shape[0]


        
# This function creates a custom output layer for the neural network 
class OutputLayer( Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dims = output_dims
        self.a = tf.Variable(np.zeros((1,self.output_dims[0],self.output_dims[1],1)), dtype=tf.dtypes.float32,trainable=False)
        super(OutputLayer, self).__init__(**kwargs)

    def call(self, x):
        self.a.assign(tf.multiply(tf.cast(tf.divide(tf.reduce_sum(x[0]),tf.reduce_sum(x[1])), tf.float32),x[2]))
        #print(self.a.numpy())
        return tf.multiply(tf.cast(tf.divide(tf.reduce_sum(x[0]),tf.reduce_sum(x[1])), tf.float32),x[2])

    def compute_output_shape(self, input_shape):
        return (self.output_dims)


# This function creates a custom layer to see TF 
class LayerBis( Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dims = output_dims
        self.a = tf.Variable(np.zeros((1,self.output_dims[1],self.output_dims[2],1)), dtype=tf.dtypes.float32,trainable=False)
        super(LayerBis, self).__init__(**kwargs)

    def call(self, x):
        self.a.assign(tf.abs(tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64)))
        #print(self.a.numpy())
        return tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64)
    
    def compute_output_shape(self, input_shape):
        return (self.output_dims)


# Custom layer created to optimize the LED positions
class pos_cal(Layer):
    def __init__(self, **kwargs):
        self.i = 0
        super(pos_cal, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lambd2 = self.add_weight(name='lambda2',
                                     shape=((1,1)), # Adding one dimension for broadcasting
                                     initializer='zeros',# Try also 'ones' and 'uniform'
                                     trainable=True
                                     )
                                     
        super(pos_cal, self).build(input_shape)

    def call(self, x):
        a = self.lambd2
        a = tf.keras.backend.cast(a, dtype='float32')
        #b = self.lambd2[1,0]
        new_pos = tf.math.add(a, x, name=None)

        return new_pos

    def compute_output_shape(self, input_shape):
        return input_shape

#class pos_angle(Layer):
    # def __init__(self, **kwargs):
    #     self.i = 0
    #     super(pos_angle, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     self.lambd2 = self.add_weight(name='lambda2',
    #                                  shape=((1,1)), # Adding one dimension for broadcasting
    #                                  initializer='zeros',# Try also 'ones' and 'uniform'
    #                                  trainable=True
    #                                  )
                                     
    #     super(pos_angle, self).build(input_shape)

    # def call(self, x,y):
    #     a = self.lambd2
    #     a = tf.keras.backend.cast(a, dtype='float32')
    #     #b = self.lambd2[1,0]
    #     new_pos_x = x*tf.math.cos(a) + y*tf.math.sin(a)
    #     new_pos_y = -x*tf.math.sin(a) + y*tf.math.cos(a)
    #     return new_pos_x,new_pos_y

    # def compute_output_shape(self, input_shape):
    #     return input_shape


import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Layer, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, Add, Subtract, Cropping2D, Reshape, Dense
from tensorflow.keras.models import Model  # C'est ici qu'il faut importer Model
from tensorflow.keras import initializers, constraints

class pos_angle(Layer):
    def __init__(self, **kwargs):
        super(pos_angle, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lambd2 = self.add_weight(name='lambda2',
                                      shape=(1, 1),
                                      initializer='zeros',
                                      trainable=True)
        super(pos_angle, self).build(input_shape)

    def call(self, inputs):
        # Assurez-vous que `inputs` est une liste ou un tuple contenant x et y
        x, y = inputs
        a = tf.cast(self.lambd2, dtype='float32')
        new_pos_x = x * tf.math.cos(a) + y * tf.math.sin(a)
        new_pos_y = -x * tf.math.sin(a) + y * tf.math.cos(a)
        return [new_pos_x, new_pos_y]

class pos_distance(Layer):
    def __init__(self, **kwargs):
        self.i = 0
        super(pos_distance, self).__init__(**kwargs)

    def build(self, input_shape):
        self.lambd2 = self.add_weight(name='lambda2',
                                     shape=((2,1)), # Adding one dimension for broadcasting
                                     initializer='ones',# Try also 'ones' and 'uniform'
                                     trainable=True
                                     )
                                     
        super(pos_distance, self).build(input_shape)

    def call(self, x,y):
        a = self.lambd2[0,0]
        a = tf.keras.backend.cast(a, dtype='float32')
        b = self.lambd2[1,0]
        new_pos_x = tf.math.multiply(x,a) 
        new_pos_y = tf.math.multiply(y,b) 
        return new_pos_x,new_pos_y

    def compute_output_shape(self, input_shape):
        return input_shape
    
    