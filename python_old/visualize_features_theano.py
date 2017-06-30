#####################################################################
# Generates visualizations of feature maps created CONV layers.
# It loads network weights from a pickled binary file. This is also
# a quick implementation with hard coding and lack of flexibility.
#
# Adapted from goo.gl/q7LazM
#####################################################################

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser(description='visualize params')
parser.add_argument("params_file_path", help="weights file")
parser.add_argument("layers_to_visualize", nargs='+', type=int,
                    help="list of integers corresponding to layers to visualize")
args = parser.parse_args()
params_file_path = args.params_file_path
layers_to_visualize = args.layers_to_visualize

params = pickle.load(open(params_file_path, "rb"), encoding="latin1")
for layer in layers_to_visualize:
    conv_filter = params[layer]
    print(conv_filter.shape)
    features = conv_filter.shape[0]
    channels = conv_filter.shape[1]
    dim = [conv_filter.shape[2], conv_filter.shape[3]]
    w_max, w_min = [], []
    for i in range(channels):
        w_max.append(np.max(conv_filter[:,i,:,:]))
        w_min.append(np.min(conv_filter[:,i,:,:]))
    n_rows = math.ceil(np.sqrt(features))
    n_cols = math.ceil(np.sqrt(features))

    if layer == 0:
        fig_red, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        for feat, ax in zip(range(features), axes.flat):
            feature = conv_filter[feat, 0].reshape(dim[0], dim[1])
            im = ax.imshow(feature, cmap='Reds', vmin=w_min[0], vmax=w_max[0])
            ax.axis('off')
        for i in range(n_rows * n_cols - features):
            axes[-1, -1 - i].axis('off')
        cax = fig_red.add_axes([0.9, 0.6, 0.02, 0.3])
        plt.colorbar(im, cax=cax, ticks=[w_min[0], w_max[0]])

        fig_green, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        for feat, ax in zip(range(features), axes.flat):
            feature = conv_filter[feat, 1].reshape(dim[0], dim[1])
            im = ax.imshow(feature, cmap='Greens', vmin=w_min[1], vmax=w_max[1])
            ax.axis('off')
        for i in range(n_rows * n_cols - features):
            axes[-1, -1 - i].axis('off')
        cax = fig_green.add_axes([0.9, 0.6, 0.02, 0.3])
        plt.colorbar(im, cax=cax, ticks=[w_min[1], w_max[1]])

        fig_blue, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        for feat, ax in zip(range(features), axes.flat):
            feature = conv_filter[feat, 2].reshape(dim[0], dim[1])
            im = ax.imshow(feature, cmap='Blues', vmin=w_min[2], vmax=w_max[2])
            ax.axis('off')
        for i in range(n_rows * n_cols - features):
            axes[-1, -1 - i].axis('off')
        cax = fig_blue.add_axes([0.9, 0.6, 0.02, 0.3])
        plt.colorbar(im, cax=cax, ticks=[w_min[2], w_max[2]])

        norm = max([abs(m) for m in w_max] + [abs(n) for n in w_min])
        conv_filter_norm = conv_filter / norm * 127.5 + 127.5
        fig_brg, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        for feat, ax in zip(range(features), axes.flat):
            feature = conv_filter_norm[feat, :].reshape(3, dim[0], dim[1])
            feature_rgb = np.zeros([3, dim[0], dim[1]])
            feature_rgb[0] = feature[0, :, :]
            feature_rgb[1] = feature[1, :, :]
            feature_rgb[2] = feature[2, :, :]
            im = ax.imshow(np.transpose(feature_rgb, [1, 2, 0]))
            ax.axis('off')
        for i in range(n_rows * n_cols - features):
            axes[-1, -1 - i].axis('off')
        #cax = fig_brg.add_axes([0.9, 0.1, 0.03, 0.8])
        #plt.colorbar(im, cax=cax)
    
    else:
        conv_slice = 1
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        for feat, ax in zip(range(features), axes.flat):
            feature = conv_filter[feat, conv_slice].reshape(dim[0], dim[1])
            im = ax.imshow(feature, cmap='gray', 
                           vmin=w_min[conv_slice], vmax=w_max[conv_slice])
            ax.axis('off')
        for i in range(n_rows * n_cols - features):
            axes[-1, -1 - i].axis('off')
        cax = fig.add_axes([0.9, 0.6, 0.02, 0.3])
        plt.colorbar(im, cax=cax, ticks=[w_min[conv_slice], w_max[conv_slice]])

plt.show()
