# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:train_keras.py
# software: PyCharm

import tensorflow.keras as keras
from retinanet import retinanet
from config.configs import config
from dataset.get_dataset import DataGenerator
from core.loss import retina_loss


if __name__ == '__main__':

    inputs = keras.Input(shape=(416, 416, 3))
    retina_model = retinanet(inputs=inputs,
                             out_channels=256,
                             num_classes=6,
                             num_anchors=9)
    retina_model.load_weights('./datas/resnet50_coco_best_v2.1.0.h5', by_name=True, skip_mismatch=True)
    print('load weights successfully!!')

    outputs = retina_model.outputs
    y_true = [keras.Input(shape=(None, 5)),
              keras.Input(shape=(None, 7))]
    loss_input = [y_true, outputs]
    model_loss = keras.layers.Lambda(retina_loss,
                                     output_shape=(1,),
                                     name='retina_loss')(loss_input)
    model = keras.Model([inputs, y_true], model_loss)

    train_txt = config.TRAIN.TRAIN_TXT
    test_txt = config.TRAIN.TEST_TXT
    class_txt = config.TRAIN.CLASS_TXT
    batch_size1 = config.TRAIN.BATCH_SIZE1
    batch_size2 = config.TRAIN.BATCH_SIZE2
    input_shape = config.TRAIN.INPUT_SHAPE

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

    # ####################################################
    # data generator
    data_gene_1 = DataGenerator(anno_lines=train_anno,
                                input_shape=input_shape,
                                num_classes=num_classes,
                                batch_size=batch_size1)
    generate_data_1 = data_gene_1.data_generate()
    # ####################################################

    optimizer1 = keras.optimizers.Adam(learning_rate=0.001)
    # freeze the first 174 layers
    # training in stage 1
    num_freeze_layers = config.TRAIN.FREEZE
    for i in range(num_freeze_layers):
        retina_model.layers[i].trainable = False
    print('have frozen resnet model and start training')

    model.compile(optimizer=optimizer1,
                  loss={'retina_loss': lambda y_true, y_pred: y_pred})

    model.fit_generator(generator=generate_data_1,
                        steps_per_epoch=train_steps_1,
                        epochs=50)
