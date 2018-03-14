# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A WaveNet-style AutoEncoder Configuration and FastGeneration Config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
from magenta.models.nsynth import reader
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import masked


class FastGenerationConfig(object):
  """Configuration object that helps manage the graph."""

  def __init__(self, batch_size=1):
    """."""
    self.batch_size = batch_size

  def build(self, inputs):
    """Build the graph for this configuration.

    Args:
      inputs: A dict of inputs. For training, should contain 'wav'.

    Returns:
      A dict of outputs that includes the 'predictions',
      'init_ops', the 'push_ops', and the 'quantized_input'.
    """
    num_stages = 10
    num_layers = 30
    filter_length = 3
    width = 512
    skip_width = 256
    num_z = 16

    # Encode the source with 8-bit Mu-Law.
    x = inputs['wav']
    batch_size = self.batch_size
    x_quantized = utils.mu_law(x)
    x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
    x_scaled = tf.expand_dims(x_scaled, 2)

    encoding = tf.placeholder(
        name='encoding', shape=[batch_size, num_z], dtype=tf.float32)
    en = tf.expand_dims(encoding, 1)

    init_ops, push_ops = [], []

    ###
    # The WaveNet Decoder.
    ###
    l = x_scaled
    l, inits, pushs = utils.causal_linear(
        x=l,
        n_inputs=1,
        n_outputs=width,
        name='startconv',
        rate=1,
        batch_size=batch_size,
        filter_length=filter_length)

    for init in inits:
      init_ops.append(init)
    for push in pushs:
      push_ops.append(push)

    # Set up skip connections.
    s = utils.linear(l, width, skip_width, name='skip_start')

    # Residual blocks with skip connections.
    for i in range(num_layers):
      dilation = 2**(i % num_stages)

      # dilated masked cnn
      d, inits, pushs = utils.causal_linear(
          x=l,
          n_inputs=width,
          n_outputs=width * 2,
          name='dilatedconv_%d' % (i + 1),
          rate=dilation,
          batch_size=batch_size,
          filter_length=filter_length)

      for init in inits:
        init_ops.append(init)
      for push in pushs:
        push_ops.append(push)

      # local conditioning
      d += utils.linear(en, num_z, width * 2, name='cond_map_%d' % (i + 1))

      # gated cnn
      assert d.get_shape().as_list()[2] % 2 == 0
      m = d.get_shape().as_list()[2] // 2
      d = tf.sigmoid(d[:, :, :m]) * tf.tanh(d[:, :, m:])

      # residuals
      l += utils.linear(d, width, width, name='res_%d' % (i + 1))

      # skips
      s += utils.linear(d, width, skip_width, name='skip_%d' % (i + 1))

    s = tf.nn.relu(s)
    s = (utils.linear(s, skip_width, skip_width, name='out1') + utils.linear(
        en, num_z, skip_width, name='cond_map_out1'))
    s = tf.nn.relu(s)

    ###
    # Compute the logits and get the loss.
    ###
    logits = utils.linear(s, skip_width, 256, name='logits')
    logits = tf.reshape(logits, [-1, 256])
    probs = tf.nn.softmax(logits, name='softmax')

    return {
        'init_ops': init_ops,
        'push_ops': push_ops,
        'predictions': probs,
        'encoding': encoding,
        'quantized_input': x_quantized,
    }


class Config(object):
  """Configuration object that helps manage the graph."""

  def __init__(self, train_path=None):
    self.num_iters = 200000
    #learning rate is bigger at begining, then smaller
    self.learning_rate_schedule = {
        0: 2e-4,
        90000: 4e-4 / 3,
        120000: 6e-5,
        150000: 4e-5,
        180000: 2e-5,
        210000: 6e-6,
        240000: 2e-6,
    }
    self.ae_hop_length = 512
    self.ae_bottleneck_width = 16
    self.train_path = train_path

  def get_batch(self, batch_size,wav_piece_length):
    assert self.train_path is not None
    data_train = reader.MyDataset(self.train_path, is_training=True, wav_piece_length=wav_piece_length)
    return data_train.get_wavenet_batch(batch_size, length=6144)

  @staticmethod
  def _condition(x, encoding):
    """Condition the input on the encoding.

    Args:
      x: The [mb, length, channels] float tensor input.
      encoding: The [mb, encoding_length, channels] float tensor encoding.

    Returns:
      The output after broadcasting the encoding to x's shape and adding them.
    """
    mb, length, channels = x.get_shape().as_list()
    enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
    assert enc_mb == mb
    assert enc_channels == channels

    encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
    x = tf.reshape(x, [mb, enc_length, -1, channels])
    x += encoding
    x = tf.reshape(x, [mb, length, channels])
    x.set_shape([mb, length, channels])
    return x

  def build(self, inputs, is_training):
    """Build the graph for this configuration.

    Args:
      inputs: A dict of inputs. For training, should contain 'wav'.
      is_training: Whether we are training or not. Not used in this config.

    Returns:
      A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
      the 'quantized_input', and whatever metrics we want to track for eval.
    """
    del is_training
    num_stages = 10
    num_layers = 30
    filter_length = 3
    width = 512
    skip_width = 256
    ae_num_stages = 10
    ae_num_layers = 30
    ae_filter_length = 3
    ae_width = 128

    print("@build, inputs: ", inputs) #pitch shape=(1,),  wav shape=(1, 6144), key shape=(1,) 
    # Encode the source with 8-bit Mu-Law.
    x = inputs['wav']
    print("@build, x: ",x) #shape=(1, 6144)
    x_quantized = utils.mu_law(x)
    print("@build, x_quantized: ",x_quantized) #shape=(1, 6144)
    x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
    print("@build, x_scaled@1: ",x_scaled) #shape=(1, 6144)
    x_scaled = tf.expand_dims(x_scaled, 2)
    print("@build, x_scaled@2: ",x_scaled) #shape=(1, 6144, 1)

    ###
    # The Non-Causal Temporal Encoder.
    ###
    print("@build, ##Non-Causal Temporal Encoder...")
    print("\t create Layer ae_startconv")
    print("\t input[x_scaled] is: ",x_scaled)
    en = masked.conv1d(
        x_scaled,
        causal=False,
        num_filters=ae_width, #ae_width = 128
        filter_length=ae_filter_length,
        name='ae_startconv')
    print("\t ae_startconv output [en] is:", en) #shape=(1. 6144, 128)
    print("\t create Layer ae_startconv Done\n")

    for num_layer in range(ae_num_layers):
      dilation = 2**(num_layer % ae_num_stages)
      print("\t create Layer relu")
      print("\t input[en] is: ",en) #shape=(1. 6144, 128)
      d = tf.nn.relu(en)
      print("\t relu output [d] is:", d)
      print("\t create Layer relu Done\n")
      
      print("\t create Layer ae_dilatedconv_{}, dilation={}".format(num_layer+1,dilation))
      print("\t input[d] is: ",d)
      d = masked.conv1d(
          d,
          causal=False,
          num_filters=ae_width, #128
          filter_length=ae_filter_length,
          dilation=dilation,
          name='ae_dilatedconv_%d' % (num_layer + 1))
      print("\t output [d] is:", d)
      print("\t create Layer ae_dilatedconv_{}, dilation={} Done\n".format(num_layer+1,dilation))
      
      print("\t create Layer relu")
      print("\t input[d] is: ",d)
      d = tf.nn.relu(d)
      print("\t relu output [d] is:", d)
      print("\t create Layer relu Done\n")
      
      print("\t create Layer ae_res_{}".format(num_layer+1))
      print("\t input[en] is: ",en)
      print("\t input[d] is: ",d)
      en += masked.conv1d(
          d,
          num_filters=ae_width, #128
          filter_length=1,
          name='ae_res_%d' % (num_layer + 1))
      print("\t output [en] is:", en) #shape=(1, 6144, 128)
      print("\t create Layer ae_res_{} Done\n".format(num_layer+1))

    print("\t create Layer ae_bottleneck")
    print("\t input[en] is: ",en) #shape=(1, 6144, 128)
    en = masked.conv1d(
        en,
        num_filters=self.ae_bottleneck_width, #16
        filter_length=1,
        name='ae_bottleneck')
    print("\t output[en] is: ",en) #shape=(1, 6144, 16)
    print("\t create Layer ae_bottleneck Done\n")

    print("\t create ae_pool")
    print("\t input[en] is: ",en) #shape=(1, 6144, 16)
    en = masked.pool1d(en, self.ae_hop_length, name='ae_pool', mode='avg') #ae_hop_length=512
    print("\t output[en] is: ",en) #shape=(1, 12, 16) #6144/512=12
    print("\t create ae_pool Done\n")
    
    encoding = en #encoding is 'feature vector', (125,16) for every 4 seconds voice. 125=4x16000/512
    print("\t ##Non-Causal Temporal Encoder output[en|encoding] is: ",encoding)
    print("@build, ##Non-Causal Temporal Encoder...Done\n")
    
    ###
    # The WaveNet Decoder.
    ###
    print("@build, ##The WaveNet Decoder...")
    print("\t input[x_scaled] is: ",x_scaled) #shape=(1, 6144, 1)
    l = masked.shift_right(x_scaled)
    print("\t create startconv")
    print("\t input[l] is: ",l) #shape=(1, 6144, 1)
    l = masked.conv1d(
        l, num_filters=width, filter_length=filter_length, name='startconv') #width=512
    print("\t output[l] is: ",l)  #shape=(1, 6144, 512)
    print("\t create startconv Done\n")

    # Set up skip connections.
    print("\t create skip_start")
    print("\t input[l] is: ",l)
    s = masked.conv1d(
        l, num_filters=skip_width, filter_length=1, name='skip_start') #skip_width=256
    print("\t output[s] is: ",s)  #shape=(1, 6144, 256)
    print("\t create skip_start Done\n")

    # Residual blocks with skip connections.
    for i in range(num_layers):
      dilation = 2**(i % num_stages)
      print("\t create dilatedconv_{}, dilation={}".format(i+1,dilation))
      print("\t input[l] is: ",l)
      d = masked.conv1d(
          l,
          num_filters=2 * width,
          filter_length=filter_length,
          dilation=dilation,
          name='dilatedconv_%d' % (i + 1))
      print("\t output[d] is: ",d)   #shape=(1, 6144, 1024)
      print("\t create dilatedconv_{}, dilation={} Done\n".format(i+1,dilation))
      
      print("\t create _condition for cond_map_{}".format(i+1))
      print("\t input[d] is: ",d)
      print("\t input[en] is: ",en)
      d = self._condition(d,
                          masked.conv1d(
                              en,
                              num_filters=2 * width,
                              filter_length=1,
                              name='cond_map_%d' % (i + 1)))
      print("\t output[d] is: ",d)
      print("\t create _condition for cond_map_{} Done\n".format(i+1))
      
      assert d.get_shape().as_list()[2] % 2 == 0
      m = d.get_shape().as_list()[2] // 2
      d_sigmoid = tf.sigmoid(d[:, :, :m])
      d_tanh = tf.tanh(d[:, :, m:])
      d = d_sigmoid * d_tanh
      print("\t d after some cacule:",d) #shape=(1, 6144, 512)
      print("")

      print("\t create res_{}".format(i+1))
      print("\t input[d] is: ",d) #shape=(1, 6144, 512)
      print("\t input[l] is: ",l) #shape=(1, 6144, 512)
      l += masked.conv1d(
          d, num_filters=width, filter_length=1, name='res_%d' % (i + 1)) #width=512
      print("\t output[l] is: ",l) #shape=(1, 6144, 512)
      print("\t create res_{} Done\n".format(i+1))
      
      print("\t create skip_{}".format(i+1))
      print("\t input[d] is: ",d) #shape=(1, 6144, 512)
      print("\t input[s] is: ",s) #shape=(1, 6144, 256)
      s += masked.conv1d(
          d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1)) #skip_width=256
      print("\t output[s] is: ",s) #shape=(1, 6144, 256)
      print("\t create skip_{} Done\n".format(i+1))

    print("\t create Layer relu")
    print("\t input[s] is: ",s) #shape=(1, 6144, 256)
    s = tf.nn.relu(s) 
    print("\t output[s] is: ",s) #shape=(1, 6144, 256) 
    print("\t create Layer relu Done\n")
    
    print("\t create Layer out1")
    print("\t input[s] is: ",s) #shape=(1, 6144, 256)
    s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1') #skip_width=256
    print("\t output[s] is: ",s) #shape=(1, 6144, 256)
    print("\t create Layer out1 Done\n")
    
    print("\t create _condition for cond_map_out1")
    print("\t input[s] is: ",s) #shape=(1, 6144, 256)
    print("\t input[en] is: ",en)
    s = self._condition(s,
                        masked.conv1d(
                            en,
                            num_filters=skip_width,  #skip_width=256
                            filter_length=1,
                            name='cond_map_out1'))
    print("\t output[s] is: ",s)
    print("\t create _condition for cond_map_out1 Done\n")

    print("\t create Layer relu")
    print("\t input[s] is: ",s) #shape=(1, 6144, 256)
    s = tf.nn.relu(s)
    print("\t output[s] is: ",s) #shape=(1, 6144, 256)
    print("\t create Layer relu Done\n")
    print("@build, ##The WaveNet Decoder...Done")

    ###
    # Compute the logits and get the loss.
    ###
    print("@build, ##Compute the logits and get the loss...")
    print("\t input[s] is: ",s)  #shape=(1, 6144, 256)
    logits = masked.conv1d(s, num_filters=256, filter_length=1, name='logits')
    print("\t output[logits] is: ",logits) #shape=(1, 6144, 256)
    logits = tf.reshape(logits, [-1, 256])
    print("\t logits after reshape: ",logits) #shape=(6144, 256) 
    probs = tf.nn.softmax(logits, name='softmax')
    print("\t probs: ",probs) #shape=(6144, 256)
    print("\t x_quantized: ",x_quantized) #
    x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
    print("\t x_indices",x_indices) #shape=(6144,)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=x_indices, name='nll'),
        0,
        name='loss')
    print("@build, ##Compute the logits and get the loss...Done")
    
    print("@build, Done, return:")
    print("\t probs:", probs) #shape=(6144, 256)
    print("\t loss:", loss) #shape=()
    print("\t x_quantized:", x_quantized) #shape=(1, 6144)
    print("\t encoding:", encoding) #shape=(1, 12, 16)

    return {
        'predictions': probs,
        'loss': loss,
        'eval': {
            'nll': loss
        },
        'quantized_input': x_quantized,
        'encoding': encoding,
    }
