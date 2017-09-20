
"""functions used to construct different architectures  
"""

import tensorflow as tf
import numpy as np
#
# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_float('weight_decay', 0.0005,
#                           """weight decay factor""")
# tf.app.flags.DEFINE_float('weight_init', 0.1,
#                             """weight init for biasis""")
# tf.app.flags.DEFINE_float('leaky_alpha', 0.1,
#                             """factor for leaky relu""")

class BasicNet(object):
  weight_decay = 5*1e-6
  weight_init = 0.1 #weight init for biasis
  leaky_alpha = 0.1
  is_training = False
  def __init__(self):
    self.pretrain_var_collection = []
    self.initial_var_collection = []
    self.trainable_var_collection = []
    self.var_rename = {}
    # self.weight_decay = FLAGS.weight_decay
    # self.weight_init = FLAGS.weight_init
    # self.leaky_alpha = FLAGS.leaky_alpha

  def leaky_relu(self, x, alpha, dtype=tf.float32):
    """leaky relu
    if x > 0:
      return x
    else:
      return alpha * x
    Args:
      x : Tensor
      alpha: float
    Return:
      y : Tensor
    """
    x = tf.cast(x, dtype=dtype)
    bool_mask = (x > 0)
    mask = tf.cast(bool_mask, dtype=dtype)
    return 1.0 * mask * x + alpha * (1 - mask) * x

  def get_bilinear(self, f_shape):
        width = f_shape[1]
        heigh = f_shape[0]
        f = width//2 + 1
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        bilinear = bilinear / (np.sum(bilinear)*f_shape[2])
        for i in range(f_shape[2]):
          for j in range(f_shape[3]):
            weights[:, :, i, j] = bilinear


        return weights

  def get_centermask(self,f_shape): # shape[batchsize, height, width, channals]
    width = f_shape[2]
    heigh = f_shape[1]
    midw = width//2
    midh = heigh//2
    distmatrix = np.zeros([heigh, width])
    for x in range(width):
      for y in range(heigh):
        value = np.sqrt((x - midw)**2+(y - midh)**2)
        distmatrix[x, y] = value
    distmatrix = distmatrix / np.max(distmatrix)
    distmatrix = 1 -  distmatrix
    distmatrix = distmatrix[np.newaxis,...,np.newaxis]
    # distmatrix = tf.expand_dims(distmatrix, 0)
    # distmatrix = tf.expand_dims(distmatrix, 3)
    # for a in range(f_shape[0]):
    #   for b in range(f_shape[3]):
    #     mask[a, :, :, b] = distmatrix
    return distmatrix


  def _activation_summary(self,x, name = None):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    if name is None:
      tensor_name = x.op.name
    else:
      tensor_name = name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

  def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
      name = var.op.name
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name + '/mean', mean)
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar(name + '/sttdev', stddev)
        l2norm = tf.sqrt(tf.reduce_sum(tf.square(var)))
        tf.summary.scalar(name + '/l2norm', l2norm)
        tf.summary.histogram(name, var)



  # def _variable_summary(self,var):
  #   """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  #   varriable_name=var.op.name
  #   mean = tf.reduce_mean(var)
  #   tf.summary.scalar(varriable_name+'/mean', mean)
  #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
  #   tf.summary.scalar(varriable_name+'/stddev', stddev)
  #   l2norm = tf.sqrt(tf.reduce_sum(tf.square(var)))
  #   tf.summary.scalar(varriable_name + '/l2norm', l2norm)
  #   tf.summary.histogram(varriable_name+'/histogram', var)

  def _variable_on_cpu(self,name, shape, initializer, pretrain = False, trainable = True):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
      #self.var_rename['inference/' + var.op.name] = var #for translate
      # print(var.op.name)
    if  tf.get_variable_scope().reuse == False:
        if pretrain:
          self.pretrain_var_collection.append(var)
        else:
          self.initial_var_collection.append(var)
        if trainable:
          self.trainable_var_collection.append(var)

    return var




  def _variable_with_weight_decay(self,name, shape, wd, pretrain = False, bilinear = False, trainable = True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """

    if bilinear:
      weights = self.get_bilinear(shape)
      #print(weights[:,:,1,1])
      initializer = tf.constant_initializer(value=weights, dtype=tf.float32)
    else:
      initializer = tf.contrib.layers.xavier_initializer()
    var = self._variable_on_cpu(name, shape, initializer, pretrain, trainable)

    if wd and not tf.get_variable_scope().reuse:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      #weight_decay = tf.reduce_mean((var**2)*wd, name='weight_loss')
      weight_decay.set_shape([])
      tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
    return var

  def conv_layer(self,scope_name,inputs, kernel_size,num_features, stride=1, linear = False, pretrain = False, batchnormalization = False, trainable = True):
    """convolutional layer
    Args:
    input: 4 - D
    tensor[batch_size, height, width, depth]
    scope: variable_scope
    name
    kernel_size: [k_height, k_width]
    stride: int32

  Return:
  output: 4 - D
  tensor[batch_size, height / stride, width / stride, out_channels]
  """
    with tf.variable_scope(scope_name) as scope:
      input_channels = inputs.get_shape()[3].value
      weights = self._variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features], wd=self.weight_decay, pretrain = pretrain, trainable = trainable)
      biases = self._variable_on_cpu('biases',[num_features],tf.constant_initializer(self.weight_init), pretrain, trainable)
      pad_size = kernel_size // 2
      pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
      inputs_pad = tf.pad(inputs, pad_mat)
      conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID')
      self.testvar = biases
      conv_biased = tf.nn.bias_add(conv, biases,  name='linearout')
      if batchnormalization:
        conv_biased = tf.layers.batch_normalization(conv_biased, training = self.is_training)
      if linear:
        return conv_biased
      conv_rect = self.leaky_relu(conv_biased,self.leaky_alpha )
      # scope.reuse_variables()
      return conv_rect

  def transpose_conv_layer(self,scope_name,inputs, kernel_size,num_features, stride, linear = False, pretrain = False, trainable = True):
    #Filter size:A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels],different from conv.
    with tf.variable_scope(scope_name) as scope:
      input_channels = inputs.get_shape()[3].value
      weights = self._variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], wd=self.weight_decay, pretrain = pretrain, bilinear = False, trainable = trainable)
      biases = self._variable_on_cpu('biases',[num_features],tf.constant_initializer(self.weight_init), pretrain, trainable)
      # scope.reuse_variables()
      batch_size = tf.shape(inputs)[0]
      output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features])
      conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases, name='linearout')
      if linear:
        return conv_biased
      conv_rect = self.leaky_relu(conv_biased,self.leaky_alpha )
      return conv_rect

  def max_pool(self,scope_name, input, kernel_size, stride):
    """max_pool layer
    Args:
      input: 4-D tensor [batch_zie, height, width, depth]
      kernel_size: [k_height, k_width]
      stride: int32
    Return:
      output: 4-D tensor [batch_size, height/stride, width/stride, depth]
     """
    with tf.variable_scope(scope_name) as scope:
      pool = tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME',name='pooling')
    return pool


  def fc_layer(self,scope_name,inputs, hiddens, flat = False, linear = False, pretrain = False, trainable = True):
    with tf.variable_scope(scope_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_processed = tf.reshape(inputs, [-1,dim])
      else:
        dim = input_shape[1]
        inputs_processed = inputs

      weights = self._variable_with_weight_decay('weights', shape=[dim,hiddens], wd=self.weight_decay, pretrain=pretrain, trainable = trainable)
      biases = self._variable_on_cpu('biases', [hiddens], tf.constant_initializer(self.weight_init), pretrain, trainable)
      # scope.reuse_variables()
      ip = tf.add(tf.matmul(inputs_processed, weights), biases, name='linearout')
      if linear:
       return ip
      fc_relu =  self.leaky_relu(ip,self.leaky_alpha )
      return fc_relu

  def leaky_conv(self, net_in, n_filter, filter_size, strides, name, pretrain=True, trainable=True):
    return self.conv_layer(scope_name=name, inputs=net_in, kernel_size=filter_size, num_features=n_filter,
                               stride=strides, linear=False, pretrain=pretrain,
                               batchnormalization=False, trainable=trainable)

  def leaky_deconv(self, name, input_layer, n_filter, out_size):
    return self.transpose_conv_layer(scope_name=name, inputs=input_layer, kernel_size=4, num_features=n_filter,
                                         stride=2, linear=False, pretrain=True, trainable=True)

  def upsample(self, name, input_layer, out_size):
    return self.transpose_conv_layer(scope_name=name, inputs=input_layer, kernel_size=4, num_features=2,
                                         stride=2, linear=True, pretrain=True, trainable=True)

  def flow(self, name, input_layer, filter_size=3):
        return self.conv_layer(scope_name=name, inputs=input_layer, kernel_size=filter_size, num_features=2,
                               stride=1, linear=True, pretrain=True, batchnormalization=False, trainable=True)

  def conv_mask(self, net_in, mask):
      tempsize = net_in.get_shape().as_list()
      net_in_mask = tf.image.resize_images(mask, [tempsize[1], tempsize[2]])
      #print(net_in_mask.get_shape().as_list())
      return net_in * net_in_mask