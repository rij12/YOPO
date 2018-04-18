import math
from copy import deepcopy

import tensorflow as tf
import tensorflow.contrib.slim as slim

from darkflow.net.yopo.calulating_IOU import intersection_over_union, Rectangle


def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta Ground Truth
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    S, B, C = m['side'], m['num'], m['classes']

    print("Number of Grid Cells", S)

    SS = S * S  # number of grid cells

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tside    = {}'.format(m['side']))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, SS, C]
    size2 = [None, SS, B]

    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [5])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    _image = tf.placeholder(tf.float32, [None, 2])

    # iou = tf.placeholder(tf.float32, size2)
    iou = tf.placeholder(tf.float32, size2)

    self.placeholders = {
        'probs': _probs, 'confs': _confs, 'coord': _coord, 'proid': _proid,
        'image': _image, 'iou': iou
    }

    # Extract the coordinate prediction from net.out
    coords = net_out[:, SS * (C + B):]
    # Make coords array back into a tensor.
    coords = tf.reshape(coords, [-1, SS, B, 5])

    iou = tf.py_func(calculate_iou, [_image, _coord, coords, iou], tf.float32)
    iou = tf.reshape(iou, [-1, SS, B])

    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    # Class Probs * box Confidence
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(5 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    proid = sprob * _proid

    # flatten 'em all
    probs = slim.flatten(_probs)
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)
    conid = slim.flatten(conid)
    coord = slim.flatten(_coord)
    cooid = slim.flatten(cooid)

    self.fetch += [probs, confs, conid, cooid, proid]
    true = tf.concat([probs, confs, coord], 1)

    wght = tf.concat([proid, conid, cooid], 1)
    print('Building {} loss'.format(m['model']))

    # Squared mean
    loss = tf.pow(net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)


def calculate_iou(image_tens, gt_tensor, net_out_tensor, iou):
    """
    This function is a custom opertation written in python that is to be executed in the TensorFlow graph,
    the YOPO IOU function that replaces the old function in the YOLO version one algorithm.
    Unlike the previous version it can handle angles and calculate a an IOU score when two boxes are rotated.

    :param image_tens: A Tensor that describes the image width and height, used to reverse the normalisation.
    :param gt_tensor: The Ground Truth Tensor, Contains the Ground Truth i.e where the boxes in the image.
    :param net_out_tensor: The output from the network i.e where the networks thinks the bbox's are in the image for
     each class.
    :param iou: A tensor SS by B in size.
    :return: iou tensor containing the iou scores for B bounding box for SS cells.
    """
    print("Start calculate_iou")
    image_index = 0
    for ground_truth, net_out_tensor in zip(gt_tensor, net_out_tensor):
        # todo
        ground_truth = deepcopy(ground_truth)
        net_out_tensor = deepcopy(net_out_tensor)
        cell_index = 0
        S = math.sqrt(len(ground_truth))
        image_width = image_tens[image_index][0]
        image_height = image_tens[image_index][1]
        print("\nImage", image_index, " W: ", image_width, " H: ", image_height, " S: ", S)
        for ground_truth_cell, net_out_cell in zip(ground_truth, net_out_tensor):
            # print("Ground Truth Tensor:", ground_truth, "Network out Tensor:", net_out_tensor)
            cell_box_index = 0

            empty = 5
            for c in ground_truth_cell[0]:
                if c == 0:
                    empty = empty - 1
                else:
                    break

            if empty == 0:
                cell_index = cell_index + 1
                continue

            for ground_truth_box, net_out_box in zip(ground_truth_cell, net_out_cell):

                cell_x = cell_index % S
                cell_y = math.floor(cell_index / S)

                cell_width = (image_width / S)
                cell_height = (image_height / S)

                centre_x = (cell_width * cell_x) + (ground_truth_box[0] * cell_width)
                centre_y = (cell_height * cell_y) + (ground_truth_box[1] * cell_height)

                gt_width = (ground_truth_box[2] ** 2) * image_width
                gt_height = (ground_truth_box[3] ** 2) * image_height

                gt_angle = ground_truth_box[4] * 360

                # Create ground truth Tensor
                ground_truth_rec = Rectangle(centre_x, centre_y, gt_width, gt_height, gt_angle)

                out_net_centre_x = (cell_width * cell_x) + (net_out_box[0] * cell_width)
                out_net_centre_y = (cell_height * cell_y) + (net_out_box[1] * cell_height)

                out_net_width = (net_out_box[2] ** 2) * image_width
                out_net_height = (net_out_box[3] ** 2) * image_height

                net_out_angle = net_out_box[4]
                net_out_angle = net_out_angle * 360
                print("RAW: Angle Network output", net_out_box[4])

                print("GROUND TRUTH ANGLE", gt_angle, "IOU Calculation Network", net_out_angle)

                # Create ground truth Tensor
                out_net_rec = Rectangle(out_net_centre_x, out_net_centre_y, out_net_width, out_net_height,
                                        net_out_angle)

                print("Ground Truth Rectangle", ground_truth_rec)
                print("Output Network Rectangle", out_net_rec)

                iou_val = intersection_over_union(ground_truth_rec, out_net_rec)

                # if cell_index == 51:
                print("IOU for box {}: {} \n\n".format(cell_box_index, iou_val))

                iou[image_index][cell_index][cell_box_index] = iou_val

                cell_box_index = cell_box_index + 1

            cell_index = cell_index + 1
        image_index = image_index + 1
    print("End calculate_iou")
    return iou
