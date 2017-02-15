import tensorflow as tf
from utils import *
from nn import *
from audioclass_model import *
from audioclass_data import *

# LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.1 # Initial learning rate.
# NUM_EPOCHS_PER_DECAY = 50.0 # 350.0 # Epochs after which learning rate decays.
# def optimizer(num_batches_per_epoch):
#     global_step = tf.Variable(initial_value=0, trainable=False)
#     increment_step = global_step.assign_add(1)
#     decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
#     lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                     global_step,
#                                     decay_steps,
#                                     LEARNING_RATE_DECAY_FACTOR,
#                                     staircase=True)
#     opt = tf.train.GradientDescentOptimizer(lr)
#     return increment_step, opt, lr, global_step

MAX_EPOCHS = 5.0

def optimizer():
    with tf.variable_scope("Optimizer"):
        global_step = tf.Variable(initial_value=0, trainable=False)
        increment_step = global_step.assign_add(1)
        opt = tf.train.AdamOptimizer(0.001)
        return increment_step, opt, global_step

def main(use_gpu=True, restore_if_possible=True, batch_size=30):
    with tf.device("/cpu:0"):
        # Build graph:
        image_batch, label_batch, num_examples_per_epoch = input_graph(training=True, batch_size=batch_size)
        num_batches_per_epoch = num_examples_per_epoch // batch_size
        increment_step, opt, step = optimizer()
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            correct, loss, _ = forward_propagation(image_batch, label_batch, dropout=True, train=True)
            grads = opt.compute_gradients(loss)
        with tf.control_dependencies([opt.apply_gradients(grads), increment_step]):
            train = tf.no_op(name='train')
        summaries = tf.summary.merge_all()

        # Train:
        saver = tf.train.Saver()  # For saving the model
        with tf.Session(config=tf.ConfigProto(
                log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter('tflog', sess.graph)  # For logging for TensorBoard

            # Initialize the variables (like the epoch counter).
            with tf.device("/cpu:0"): # Initialize variables on the main cpu
                sess.run(tf.global_variables_initializer())
            if restore_if_possible:
                try:
                    saver.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))
                except:
                    print("Couldn't find old model. Starting from scratch.")

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            epoch_count = None
            try:
                while ((not coord.should_stop()) and (epoch_count <= MAX_EPOCHS)):
                    _, num_correct, batch_loss, i = sess.run([train, correct, loss, step])
                    in_batch = i % num_batches_per_epoch
                    if in_batch == 0:
                        in_batch = num_batches_per_epoch
                    epoch_count = (i // num_batches_per_epoch) + 1
                    print("Epoch %d. Batch %d/%d. Acc %.3f. Loss %.2f" % (epoch_count, in_batch, num_batches_per_epoch, 100*num_correct / float(batch_size), batch_loss))
                    if in_batch + 1 == num_batches_per_epoch:
                        # Checkpoint, save the model:
                        summary = sess.run(summaries)
                        summary_writer.add_summary(summary)
                        print("Saving to %s" % SAVED_MODEL_PATH)
                        saver.save(sess, SAVED_MODEL_PATH, global_step=i)

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

if __name__ == "__main__":
    main(use_gpu=False)
