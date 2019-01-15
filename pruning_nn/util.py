"""
This file contains some utility functions to calculate hessian matrix and its inverse.

Adapted from https://github.com/csyhhu/L-OBS/blob/master/PyTorch/ImageNet/util.py
Original author: Chen Shangyu (schen025@e.ntu.edu.sg)
Adaptions made by: Paul Häusner
DOI: 1705.07565
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from datetime import datetime
import tensorflow as tf
import os
import numpy as np
import pandas as pd

from pruning_nn.network import MaskedLinearLayer


# Construct hessian inverse computing graph for woodbury method
def create_woodbury_hessian_inv_graph(input_shape, dataset_size):
    """
    This function create the hessian inverse calculation graph using Woodbury method.
    """
    hessian_inv_holder = tf.placeholder(dtype=tf.float32, shape=[input_shape, input_shape])
    input_holder = tf.placeholder(dtype=tf.float32, shape=[1, input_shape])

    # [1, 4097] [4097, 4097] [4097, 1]
    denominator = dataset_size + tf.matmul(a=tf.matmul(a=input_holder, b=hessian_inv_holder), b=input_holder,
                                           transpose_b=True)

    # ([4097, 4097] [4097, 1]) ([1, 4097] [4097, 4097])
    numerator = tf.matmul(a=tf.matmul(a=hessian_inv_holder, b=input_holder, transpose_b=True),
                          b=tf.matmul(a=input_holder, b=hessian_inv_holder))

    hessian_inv_op = hessian_inv_holder - numerator * (1.00 / denominator)

    return hessian_inv_holder, input_holder, hessian_inv_op


def generate_hessian_inv_woodbury(net, trainloader, layer_name, layer_type, n_batch_used=100, batch_size=2,
                                  stride_factor=3, use_tf_backend=True, use_cuda=True):
    """
    This function calculated Hessian inverse matrix by Woodbury matrix identity.
    Args:
        net: PyTorch model
        trainloader: PyTorch dataloader
        layer_name: Name of the layer
        layer_type: 'C' for Convolution (with bias),
                    'R' for res layer (without bias),
                    'F' for Fully-Connected (with bias).
                    I am sure you will know why the bias term is emphasized here as you are clever.
        n_batch_used: number of batches used to generate hessian.
        batch_size: Batch size. Because hessian calculation graph is quite huge. A small (like 2) number
                    of batch size if recommended here.
        stride_factor: Due to the same reason mentioned above, bigger stride results in fewer extracted
                    image patches (think about how convolution works).  stride_factor is multiplied by
                    actual stride in latter use. Therefore when stride_factor == 1, it extract patches in
                    original way. However, it may results in some GPU/CPU memory troubles. If you meet such,
                    you can increase the stride factor here.
        use_cuda: whether you can use cuda or not.
        use_tf_backend: A TensorFlow wrapper is used to accelerate the process. True for using such wrapper.
    """
    hessian_inverse = None
    dataset_size = 0
    freq_moniter = (n_batch_used * batch_size) / 50  # Total 50 times of printing information

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net.eval()
    for batch_idx, (inputs, _) in enumerate(trainloader):

        if use_cuda:
            inputs = inputs.cuda()

        net(Variable(inputs, volatile=True))

        layer_input = net.module.layer_input[layer_name]

        # Construct tf op for convolution and res layer
        if batch_idx == 0:
            if layer_type == 'C' or layer_type == 'R':
                print('[%s] Now construct patches extraction op for layer %s' % (datetime.now(), layer_name))
                layer_input_np = layer_input.permute(0, 2, 3, 1).cpu().numpy()
                layer_kernel = net.module.layer_kernel[layer_name]
                layer_stride = net.module.layer_stride[layer_name] * stride_factor
                layer_input_holder = tf.placeholder(dtype=tf.float32, shape=layer_input_np.shape)

                get_patches_op = \
                    tf.extract_image_patches(images=layer_input_holder,
                                             ksizes=[1, layer_kernel, layer_kernel, 1],
                                             strides=[1, layer_stride, layer_stride, 1],
                                             rates=[1, 1, 1, 1],
                                             padding='SAME')
                # For a convolution input, extracted pathes would be: [1, 9, 9, 2304]
                dataset_size = n_batch_used * int(get_patches_op.get_shape()[0]) * \
                               int(get_patches_op.get_shape()[1]) * int(get_patches_op.get_shape()[2])
                input_dimension = get_patches_op.get_shape()[3]
                if layer_type == 'C':
                    # In convolution layer, input dimension should be added one for bias term
                    hessian_inverse = 1000000 * np.eye(input_dimension + 1)
                    if use_tf_backend:
                        print('You choose tf backend to calculate Woodbury, constructing your graph.')
                        hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = \
                            create_woodbury_hessian_inv_graph(input_dimension + 1, dataset_size)
                else:
                    hessian_inverse = 1000000 * np.eye(input_dimension)
                    if use_tf_backend:
                        print('You choose tf backend to calculate Woodbury, constructing your graph.')
                        hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = \
                            create_woodbury_hessian_inv_graph(input_dimension, dataset_size)
            else:
                layer_input_np = layer_input.cpu().numpy()
                input_dimension = layer_input_np.shape[1]
                dataset_size = n_batch_used * batch_size
                hessian_inverse = 1000000 * np.eye(input_dimension + 1)
                if use_tf_backend:
                    print('You choose tf backend to calculate Woodbury, constructing your graph.')
                    hessian_inv_holder, input_holder, Woodbury_hessian_inv_op = create_woodbury_hessian_inv_graph(
                        input_dimension + 1, dataset_size)

            print('[%s] dataset: %d, input dimension: %d' % (datetime.now(), dataset_size, input_dimension))

        # Begin process
        if layer_type == 'F':
            this_layer_input = layer_input.cpu().numpy()  # [2, 4096]
            for i in range(this_layer_input.shape[0]):
                this_input = this_layer_input[i]
                # print this_input.shape
                # print np.array([1.0]).shape
                wb = np.concatenate([this_input.reshape(1, -1), np.array([1.0]).reshape(1, -1)], axis=1)  # [1, 4097]
                if use_tf_backend:
                    hessian_inverse = sess.run(Woodbury_hessian_inv_op, feed_dict={
                        hessian_inv_holder: hessian_inverse,
                        input_holder: wb
                    })
                else:
                    # [1, 4097] [4097, 4097] [4097, 1]
                    denominator = dataset_size + np.dot(np.dot(wb, hessian_inverse), wb.T)
                    # [4097, 4097] [4097, 1] [1, 4097] [4097, 4097]
                    numerator = np.dot(np.dot(hessian_inverse, wb.T), np.dot(wb, hessian_inverse))
                    hessian_inverse = hessian_inverse - numerator * (1.0 / denominator)

        elif layer_type == 'C' or layer_type == 'R':
            this_layer_input = layer_input.permute(0, 2, 3, 1).cpu().numpy()
            this_patch = sess.run(get_patches_op, feed_dict={layer_input_holder: this_layer_input})

            for i in range(this_patch.shape[0]):
                for j in range(this_patch.shape[1]):
                    for m in range(this_patch.shape[2]):
                        this_input = this_patch[i][j][m]
                        if layer_type == 'C':
                            wb = np.concatenate([this_input.reshape(1, -1), np.array([1.0]).reshape(1, -1)],
                                                axis=1)  # [1, 2305]
                        else:
                            wb = this_input.reshape(1, -1)  # [1, 2304]
                        if use_tf_backend:
                            hessian_inverse = sess.run(Woodbury_hessian_inv_op, feed_dict={
                                hessian_inv_holder: hessian_inverse,
                                input_holder: wb
                            })
                        else:
                            denominator = dataset_size + np.dot(np.dot(wb, hessian_inverse), wb.T)
                            numerator = np.dot(np.dot(hessian_inverse, wb.T), np.dot(wb, hessian_inverse))
                            hessian_inverse = hessian_inverse - numerator * (1.0 / denominator)

        if batch_idx % freq_moniter == 0:
            print('[%s] Now finish image No. %d / %d'
                  % (datetime.now(), batch_idx * batch_size, n_batch_used * batch_size))

        if batch_idx == n_batch_used:
            sess.close()
            break

    return hessian_inverse


def generate_hessian_inverse_fc(hessian_inverse_path, w_layer_path, layer_input_train_dir):
    """

    :param hessian_inverse_path:    Path where the hessian inverse should be deleted.
    :param w_layer_path:            The weight matrix of the layer.
    :param layer_input_train_dir:   The inputs on the layer.
    :return:
    """
    pass


"""
Normal util methods start here
"""


def prune_network_by_saliency(network, percentage):
    """
    Prune the number of percentage weights from the network. The elements are pruned according to the saliency that is
    set in the network. By default the saliency is the actual weight of the connections. The number of elements are
    pruned blinded. Meaning in each layer a different percentage of elements might get pruned but overall the given one
    is removed from the network. This is a different approach than used in the method below where we prune exactly the
    given percentage from each layer.

    :param network:     The layers of the network.
    :param percentage:  The percentage of weights that should be pruned.
    :return: The calculated threshold.
    """
    all_sal = []
    for layer in get_single_pruning_layer(network):
        # flatten both weights and mask
        mask = list(layer.get_mask().abs().numpy().flatten())
        saliency = list(layer.get_saliency().numpy().flatten())

        # zip, filter, unzip the two lists
        mask, filtered_saliency = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, saliency) if masked_val == 1))
        # add all saliencies to list
        all_sal += filtered_saliency

    # calculate percentile
    th = np.percentile(np.array(all_sal), percentage)

    # set the mask
    for layer in get_single_pruning_layer(network):
        # All deleted weights should be set to zero so they should definetly be less than the threshold since this is
        # positive.
        layer.set_mask(torch.ge(layer.get_saliency(), th).float())


def prune_layer_by_saliency(network, percentage):
    for layer in get_single_pruning_layer(network):
        mask = list(layer.get_mask().abs().numpy().flatten())
        saliency = list(layer.get_saliency().numpy().flatten())
        mask, filtered_saliency = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, saliency) if masked_val == 1))
        th = np.percentile(np.array(filtered_saliency), percentage)
        layer.set_mask(torch.ge(layer.get_saliency(), th).float())


def get_single_pruning_layer(network):
    for child in network.children():
        if type(child) == MaskedLinearLayer:
            yield child


def get_weight_distribution(network):
    all_weights = []
    for layer in get_single_pruning_layer(network):
        mask = list(layer.get_mask().numpy().flatten())
        weights = list(layer.get_weight().data.numpy().flatten())

        masked_val, filtered_weights = zip(
            *((masked_val, weight_val) for masked_val, weight_val in zip(mask, weights) if masked_val == 1))

        all_weights += list(filtered_weights)
    return pd.DataFrame(data=all_weights)


def get_network_weight_count(network):
    total_weights = 0
    for layer in get_single_pruning_layer(network):
        total_weights += layer.get_weight_count()
    return total_weights


def reset_pruned_network(network):
    for layer in get_single_pruning_layer(network):
        layer.reset_parameters(keep_mask=True)

