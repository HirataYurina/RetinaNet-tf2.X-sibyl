# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:train.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
from config.configs import config
from retinanet import RetinaNet, retinanet
from core.loss import retina_loss
from dataset.get_dataset import DataGenerator
import logging
from net.resnet import ResNet
from net.fpn import FPN
from net.subnet import class_subnet, box_subnet

# retina_model = RetinaNet(out_channels=config.MODEL.OUT_CHANNELS,
#                          num_anchors=config.MODEL.NUM_ANCHORS,
#                          num_classes=config.MODEL.NUM_CLASSES)


# training strategy can refer to "Focal Loss for Dense Object Detection"
def train_step(optimizer, input_img, y_true, batch_size, retina_model):

    with tf.GradientTape() as tape:
        y_pred = retina_model(input_img, training=True)
        losses = retina_loss(y_pred, y_true, batch_size)

    gradients = tape.gradient(losses, retina_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, retina_model.trainable_variables))
    return losses


def log(diary_path, lr1, lr2, batch_size1, batch_size2, epoch1, epoch2, input_shape, decay_mode):
    logger = logging.getLogger()
    handler = logging.FileHandler(diary_path)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info('lr1:{}'.format(lr1))
    logger.info('lr2:{}'.format(lr2))
    logger.info('batch_size1:{}'.format(batch_size1))
    logger.info('batch_size2:{}'.format(batch_size2))
    logger.info('epoch1:{}'.format(epoch1))
    logger.info('epoch2:{}'.format(epoch2))
    logger.info('input_shape:{}'.format(input_shape))
    logger.info('decay_mode:{}'.format(decay_mode))
    logger.info('=======================a beautiful dividing line===========================')


if __name__ == '__main__':

    lr1 = config.TRAIN.LR1
    lr2 = config.TRAIN.LR2
    batch_size1 = config.TRAIN.BATCH_SIZE1
    batch_size2 = config.TRAIN.BATCH_SIZE2
    epoch1 = config.TRAIN.EPOCH1
    epoch2 = config.TRAIN.EPOCH2
    train_txt = config.TRAIN.TRAIN_TXT
    test_txt = config.TRAIN.TEST_TXT
    class_txt = config.TRAIN.CLASS_TXT
    input_shape = config.TRAIN.INPUT_SHAPE
    save_interval = config.TRAIN.SAVE_INTERVAL
    save_path = config.TRAIN.SAVE_PATH
    diary_path = config.TRAIN.DIARY_PATH

    # inputs = keras.Input(shape=(input_shape[0], input_shape[1], 3))
    # outputs = retina_model(inputs)
    # retina_model = keras.Model(inputs, outputs)
    inputs = keras.Input(shape=(416, 416, 3))

    retina_model = retinanet(inputs, out_channels=256, num_classes=6, num_anchors=9)
    # retina_model.summary()
    retina_model.load_weights('./datas/resnet50_coco_best_v2.1.0.h5', by_name=True, skip_mismatch=True)
    print('load weights successfully!!')

    with open(train_txt) as f:
        train_anno = f.readlines()
    num_train = len(train_anno)
    with open(test_txt) as f:
        test_anno = f.readlines()
    num_test = len(test_anno)
    with open(class_txt) as f:
        classes = f.readlines()
    num_classes = len(classes)

    train_steps_1 = num_train // batch_size1
    train_steps_2 = num_train // batch_size2

    # ---------------------------------------------------------------------------
    # data generator
    data_gene_1 = DataGenerator(anno_lines=train_anno,
                                input_shape=input_shape,
                                num_classes=num_classes,
                                batch_size=batch_size1)
    generate_data_1 = data_gene_1.data_generate()

    # first stage
    optimizer1 = keras.optimizers.Adam(learning_rate=lr1)

    # training in stage 1
    num_freeze_layers = config.TRAIN.FREEZE
    for i in range(num_freeze_layers):
        retina_model.layers[i].trainable = False
    print('have frozen resnet model and start training')

    for i in range(epoch1):
        epoch_loss = 0
        step_counter = 0
        for data in generate_data_1:

            if step_counter == train_steps_1:
                break

            image_data = data[0]
            y_true = data[1:]
            # train one step
            losses = train_step(optimizer=optimizer1,
                                input_img=image_data,
                                y_true=y_true,
                                batch_size=batch_size1,
                                retina_model=retina_model)
            step_counter += 1
            epoch_loss += losses
            print('one step losses---', losses)
        print('epoch{}-loss:{}'.format(i + 1, epoch_loss))
        # save weights
        if step_counter % save_interval == 0:
            retina_model.save_weights(save_path + 'epoch{}-loss:{}.h5'.format(i + 1, epoch_loss))

    # ---------------------------------------------------------------------------
    # training in stage 2
    # second stage optimizer
    optimizer2 = keras.optimizers.Adam(learning_rate=lr2)
    # unfreeze total layers
    for layer in retina_model.layers:
        layer.trainable = True
    # second data generator
        # data generator
    data_gene_2 = DataGenerator(anno_lines=train_anno,
                                input_shape=input_shape,
                                num_classes=num_classes,
                                batch_size=batch_size2)
    # TODO: try to use tf.data.Dataset.from_generator().prefetch to improve GPU-Util
    generate_data_2 = data_gene_2.data_generate()

    for i in range(epoch2 - epoch1):
        step_counter = 0
        epoch_loss = 0
        for data in generate_data_2:

            if step_counter == train_steps_2:
                break

            image_data = data[0]
            y_true = data[1:]
            # train one step
            losses = train_step(optimizer=optimizer2,
                                input_img=image_data,
                                y_true=y_true,
                                batch_size=batch_size2)
            step_counter += 1
            epoch_loss += losses
        print('epoch{}-loss:{}'.format(epoch1 + i + 1, epoch_loss))
        # save weights
        if step_counter % save_interval == 0:
            retina_model.save_weights('./logs/epoch{}-loss:{}.h5'.format(epoch1 + i + 1, epoch_loss))
