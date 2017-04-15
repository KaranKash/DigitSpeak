import tensorflow as tf
from utils import *
from nn import *
from model import *
from data import *
from load_data import *
#from runeval import *
import numpy as np
from multiprocessing import Pool
from contextlib import closing


MAX_EPOCHS = 1.0

def optimizer(num_batches_per_epoch):
    with tf.variable_scope("Optimizer"):
        global_step = tf.Variable(initial_value=0, trainable=False)
        increment_step = global_step.assign_add(1)
        opt = tf.train.AdamOptimizer(0.00001)
        return increment_step, opt, global_step

#70 85
def train_network(use_gpu=True, restore_if_possible=True, english_batch=200, spanish_batch=246):
    with tf.device("/cpu:0"):
        # Build English graph:
        e_image_batch, e_label_batch, e_num_examples_per_epoch = english_input_graph(training=True, batch_size=english_batch)
        e_correct_mnist = tf.placeholder(tf.float32, shape=(english_batch, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        e_mismatch_mnist = tf.placeholder(tf.float32, shape=(english_batch, 9, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS))
        # permutation = tf.placeholder(tf.int32, shape=(batch_size))
        e_num_batches_per_epoch = e_num_examples_per_epoch // english_batch
        e_increment_step, e_opt, e_step = optimizer(e_num_batches_per_epoch)
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            # embeddings, loss, mnist = forward_propagation(image_batch, correct_mnist, mismatch_mnist, permutation, dropout=True, train=True)
            e_embeddings, e_loss, e_mnist, e_mismatch = english_forward_propagation(e_image_batch, e_correct_mnist, e_mismatch_mnist, dropout=True, train=True)
            e_grads = e_opt.compute_gradients(e_loss)
        with tf.control_dependencies([e_opt.apply_gradients(e_grads), e_increment_step]):
            e_train = tf.no_op(name='english_train')

        # Build Spanish graph:
        s_image_batch, s_label_batch, s_num_examples_per_epoch = spanish_input_graph(training=True, batch_size=spanish_batch)
        s_correct_mnist = tf.placeholder(tf.float32, shape=(spanish_batch, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        s_mismatch_mnist = tf.placeholder(tf.float32, shape=(spanish_batch, 9, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS))
        # permutation = tf.placeholder(tf.int32, shape=(batch_size))
        s_num_batches_per_epoch = s_num_examples_per_epoch // spanish_batch
        s_increment_step, s_opt, s_step = optimizer(s_num_batches_per_epoch)
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            # embeddings, loss, mnist = forward_propagation(image_batch, correct_mnist, mismatch_mnist, permutation, dropout=True, train=True)
            s_embeddings, s_loss, s_mnist, s_mismatch = spanish_forward_propagation(s_image_batch, s_correct_mnist, s_mismatch_mnist, dropout=True, train=True)
            s_grads = s_opt.compute_gradients(s_loss)
        with tf.control_dependencies([s_opt.apply_gradients(s_grads), s_increment_step]):
            s_train = tf.no_op(name='spanish_train')

        summaries = tf.summary.merge_all()

        # Train:
        with tf.Session(config=tf.ConfigProto(
                log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter('tflog', sess.graph)  # For logging for TensorBoard

            saver = tf.train.Saver()
            with tf.device("/cpu:0"):
                sess.run(tf.global_variables_initializer())

            if restore_if_possible:
                try:
                    # saver = tf.train.import_meta_graph('./model/model.ckpt-37.meta')
                    saver.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))
                    print("Found in-progress model. Will resume from there.")
                except:
                    # saver = tf.train.Saver()
                    # with tf.device("/cpu:0"):
                    #     sess.run(tf.global_variables_initializer())
                    print("Couldn't find old model. Starting from scratch.")

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            epoch_count = 1
            try:
                # training, alternate steps between spanish and english
                while ((not coord.should_stop()) and (epoch_count <= MAX_EPOCHS)):
                    # English
                    labels, spec_activations = sess.run([e_label_batch, e_embeddings])
                    labels = labels.flatten().tolist()

                    if epoch_count == MAX_EPOCHS:
                        for i in range(len(labels)):
                            spec = spec_activations[i]
                            spec = spec.reshape((1,512))
                            label = labels[i]
                            f = open("./English/" + str(label) + ".txt",'a')
                            np.savetxt(f,spec)
                            f.close()

                    mnist_batch, mismatch_mnist_batch = generate_mnist_set(labels)
                    # indices = permute_batch(labels)
                    _, batch_loss, i, mnist_set, mismatch_set = sess.run([e_train, e_loss, e_step, e_mnist, e_mismatch], feed_dict={
                        e_correct_mnist: mnist_batch, e_mismatch_mnist: mismatch_mnist_batch
                    })
                    in_batch = i % e_num_batches_per_epoch
                    if in_batch == 0:
                        in_batch = e_num_batches_per_epoch
                    epoch_count = (i // (e_num_batches_per_epoch+1)) + 1

                    acc = accuracy(labels, spec_activations, mnist_set, mismatch_set)

                    print("English. Epoch %d. Batch %d/%d. Batch Loss %.2f. Acc %.2f." % (epoch_count, in_batch, e_num_batches_per_epoch, batch_loss, acc*100))

                    # Spanish
                    labels, spec_activations = sess.run([s_label_batch, s_embeddings])
                    labels = labels.flatten().tolist()

                    if epoch_count == MAX_EPOCHS:
                        for i in range(len(labels)):
                            spec = spec_activations[i]
                            spec = spec.reshape((1,512))
                            label = labels[i]
                            f = open("./Spanish/" + str(label) + ".txt",'a')
                            np.savetxt(f,spec)
                            f.close()

                    mnist_batch, mismatch_mnist_batch = generate_mnist_set(labels)
                    # indices = permute_batch(labels)
                    _, batch_loss, i, mnist_set, mismatch_set = sess.run([s_train, s_loss, s_step, s_mnist, s_mismatch], feed_dict={
                        s_correct_mnist: mnist_batch, s_mismatch_mnist: mismatch_mnist_batch
                    })
                    in_batch = i % s_num_batches_per_epoch
                    if in_batch == 0:
                        in_batch = s_num_batches_per_epoch
                    epoch_count = (i // (s_num_batches_per_epoch+1)) + 1

                    acc = accuracy(labels, spec_activations, mnist_set, mismatch_set)

                    print("Spanish. Epoch %d. Batch %d/%d. Batch Loss %.2f. Acc %.2f." % (epoch_count, in_batch, s_num_batches_per_epoch, batch_loss, acc*100))

                    # Spanish
                    labels, spec_activations = sess.run([s_label_batch, s_embeddings])
                    labels = labels.flatten().tolist()

                    if epoch_count == MAX_EPOCHS:
                        for i in range(len(labels)):
                            spec = spec_activations[i]
                            spec = spec.reshape((1,512))
                            label = labels[i]
                            f = open("./Spanish/" + str(label) + ".txt",'a')
                            np.savetxt(f,spec)
                            f.close()

                    mnist_batch, mismatch_mnist_batch = generate_mnist_set(labels)
                    # indices = permute_batch(labels)
                    _, batch_loss, i, mnist_set, mismatch_set = sess.run([s_train, s_loss, s_step, s_mnist, s_mismatch], feed_dict={
                        s_correct_mnist: mnist_batch, s_mismatch_mnist: mismatch_mnist_batch
                    })
                    in_batch = i % s_num_batches_per_epoch
                    if in_batch == 0:
                        in_batch = s_num_batches_per_epoch
                    epoch_count = (i // (s_num_batches_per_epoch+1)) + 1

                    acc = accuracy(labels, spec_activations, mnist_set, mismatch_set)

                    print("Spanish. Epoch %d. Batch %d/%d. Batch Loss %.2f. Acc %.2f." % (epoch_count, in_batch, s_num_batches_per_epoch, batch_loss, acc*100))

                    if in_batch == s_num_batches_per_epoch:
                        # Checkpoint, save the model:
                        summary = sess.run(summaries)
                        summary_writer.add_summary(summary)
                        print("Saving to %s" % SAVED_MODEL_PATH)
                        saver.save(sess, SAVED_MODEL_PATH, global_step=i)
                        # evaluate(partition="test")

            except tf.errors.OutOfRangeError:
                print('Done running!')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

def accuracy(labels, spec_activations, mnist_set, mismatch_set):
    num_correct = 0.
    for i in range(len(labels)):
        spec = spec_activations[i].flatten()
        mnist = mnist_set[i].flatten()
        base_score = mnist.dot(spec)
        correct = True
        for mismatch in mismatch_set[i]:
            score = mismatch.flatten().dot(spec)
            if score > base_score:
                correct = False
                break
        if correct:
            num_correct += 1.
    out = num_correct/len(labels)
    return out

if __name__ == "__main__":
    train_network(use_gpu=False)
