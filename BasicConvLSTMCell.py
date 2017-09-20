
import tensorflow as tf


# import BasicNet
#
# basicnet = BasicNet.BasicNet()

class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, hiddennum, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.shape
        num_features = self.num_features
        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * hiddennum])
        return zeros


class BasicConvGRUCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
                 state_is_tuple=False, activation=tf.nn.tanh):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self.trainable_var_collection = []

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, h, mask_in, mask_h, dp_in, dp_h, scope=None):
        """Long short-term memory cell (LSTM)."""
        # Parameters of gates are concatenated into one multiply for efficiency.
        assert mask_in.get_shape().as_list()[:-1] == inputs.get_shape().as_list() and mask_h.get_shape().as_list()[
                                                                                      :-1] == inputs.get_shape().as_list()
        # print(mask_in.get_shape().as_list()[-1])
        assert mask_in.get_shape().as_list()[-1] == 3 and mask_h.get_shape().as_list()[-1] == 3



            # concat = self._conv_linear([inputs, h], self.filter_size, self.num_features * 4, True,scope = scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            # i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

        z_h = self._conv_linear([h * mask_h[..., 0]], self.filter_size, self.num_features, False,
                                scope=scope + 'h1') * (1 / dp_h)
        r_h = self._conv_linear([h * mask_h[..., 1]], self.filter_size, self.num_features, False,
                                scope=scope + 'h2') * (1 / dp_h)
        th_h = self._conv_linear([h * mask_h[..., 2]], self.filter_size, self.num_features, False,
                                scope=scope + 'h3') * (1 / dp_h)


        z_in = self._conv_linear([inputs * mask_in[..., 0]], self.filter_size, self.num_features, True,
                                 scope=scope + 'i1') * (1 / dp_in)
        r_in = self._conv_linear([inputs * mask_in[..., 1]], self.filter_size, self.num_features, True,
                                 scope=scope + 'i2') * (1 / dp_in)
        th_in = self._conv_linear([inputs * mask_in[..., 2]], self.filter_size, self.num_features, True,
                                 scope=scope + 'i3') * (1 / dp_in)

        z_t = tf.nn.sigmoid(z_h + z_in)
        r_t = tf.nn.sigmoid(r_h + r_in)
        th_t = self._activation(r_t * th_h + th_in)
        new_h = (1 - z_t) * h + z_t * th_t
        return new_h

    def _conv_linear(self, args, filter_size, num_features, bias, bias_start=0.0, scope=None):
        """convolution:
        Args:
          args: a 4D Tensor or a list of 4D, batch x n, Tensors.
          filter_size: int tuple of filter height and width.
          num_features: int, number of features.
          bias_start: starting value to initialize the bias; 0 by default.
          scope: VariableScope for the created subgraph; defaults to "Linear".
        Returns:
          A 4D Tensor with shape [batch h w num_features]
        Raises:
          ValueError: if some of the arguments has unspecified or wrong shape.
        """

        # Calculate the total size of arguments on dimension 1.
        total_arg_size_depth = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 4:
                raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
            if not shape[3]:
                raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
            else:
                total_arg_size_depth += shape[3]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope):
            matrix = tf.get_variable(
                "weights", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
            if len(args) == 1:
                res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
            else:
                res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='SAME')
            if not bias:
                return res
            bias_term = tf.get_variable(
                "biases", [num_features],
                dtype=dtype,
                initializer=tf.constant_initializer(
                    bias_start, dtype=dtype))
            if tf.get_variable_scope().reuse == False:
                self.trainable_var_collection.append(matrix)
                self.trainable_var_collection.append(bias_term)

        return res + bias_term





class BasicConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
                 state_is_tuple=False, activation=tf.nn.tanh):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self.trainable_var_collection = []

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, mask_in, mask_h, dp_in, dp_h, scope=None):
        """Long short-term memory cell (LSTM)."""
        # Parameters of gates are concatenated into one multiply for efficiency.
        assert mask_in.get_shape().as_list()[:-1] == inputs.get_shape().as_list() and mask_h.get_shape().as_list()[
                                                                                      :-1] == inputs.get_shape().as_list()
        # print(mask_in.get_shape().as_list()[-1])
        assert mask_in.get_shape().as_list()[-1] == 4 and mask_h.get_shape().as_list()[-1] == 4
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(axis=3, num_or_size_splits=2, value=state)

            # concat = self._conv_linear([inputs, h], self.filter_size, self.num_features * 4, True,scope = scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            # i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

        i_h = self._conv_linear([h * mask_h[..., 0]], self.filter_size, self.num_features, False,
                                scope=scope + 'h1') * (1 / dp_h)
        j_h = self._conv_linear([h * mask_h[..., 1]], self.filter_size, self.num_features, False,
                                scope=scope + 'h2') * (1 / dp_h)
        f_h = self._conv_linear([h * mask_h[..., 2]], self.filter_size, self.num_features, False,
                                scope=scope + 'h3') * (1 / dp_h)
        o_h = self._conv_linear([h * mask_h[..., 3]], self.filter_size, self.num_features, False,
                                scope=scope + 'h4') * (1 / dp_h)
        i_in = self._conv_linear([inputs * mask_in[..., 0]], self.filter_size, self.num_features, True,
                                 scope=scope + 'i1') * (1 / dp_in)
        j_in = self._conv_linear([inputs * mask_in[..., 1]], self.filter_size, self.num_features, True,
                                 scope=scope + 'i2') * (1 / dp_in)
        f_in = self._conv_linear([inputs * mask_in[..., 2]], self.filter_size, self.num_features, True,
                                 scope=scope + 'i3') * (1 / dp_in)
        o_in = self._conv_linear([inputs * mask_in[..., 3]], self.filter_size, self.num_features, True,
                                 scope=scope + 'i4') * (1 / dp_in)

        new_c = (c * tf.nn.sigmoid(f_h + f_in + self._forget_bias) + tf.nn.sigmoid(i_h + i_in) *
                 self._activation(j_h + j_in))
        new_h = self._activation(new_c) * tf.nn.sigmoid(o_h + o_in)

        if self._state_is_tuple:
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat(axis=3, values=[new_c, new_h])
        return new_h, new_state

    def _conv_linear(self, args, filter_size, num_features, bias, bias_start=0.0, scope=None):
        """convolution:
        Args:
          args: a 4D Tensor or a list of 4D, batch x n, Tensors.
          filter_size: int tuple of filter height and width.
          num_features: int, number of features.
          bias_start: starting value to initialize the bias; 0 by default.
          scope: VariableScope for the created subgraph; defaults to "Linear".
        Returns:
          A 4D Tensor with shape [batch h w num_features]
        Raises:
          ValueError: if some of the arguments has unspecified or wrong shape.
        """

        # Calculate the total size of arguments on dimension 1.
        total_arg_size_depth = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 4:
                raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
            if not shape[3]:
                raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
            else:
                total_arg_size_depth += shape[3]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope):
            matrix = tf.get_variable(
                "weights", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
            if len(args) == 1:
                res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
            else:
                res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='SAME')
            if not bias:
                return res
            bias_term = tf.get_variable(
                "biases", [num_features],
                dtype=dtype,
                initializer=tf.constant_initializer(
                    bias_start, dtype=dtype))
            if tf.get_variable_scope().reuse == False:
                self.trainable_var_collection.append(matrix)
                self.trainable_var_collection.append(bias_term)

        return res + bias_term
