import tensorflow as tf

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        yield tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        yield tf.summary.scalar('stddev', stddev)
        yield tf.summary.scalar('max', tf.reduce_max(var))
        yield tf.summary.scalar('min', tf.reduce_min(var))
        yield tf.summary.histogram('histogram', var)

def weight_variable(dims, wd=None):
    with tf.device("/cpu:0"):
        var = tf.get_variable("weights", dims, initializer=tf.contrib.layers.xavier_initializer())
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(dim):
    with tf.device("/cpu:0"):
        var = tf.get_variable("bias", [dim], initializer=tf.contrib.layers.xavier_initializer())
    return var
