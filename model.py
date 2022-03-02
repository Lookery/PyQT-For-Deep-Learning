import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, BatchNormalization, \
    Activation, GlobalAveragePooling1D

# GPU设置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(gpus[0]))
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 继承Layer,建立resnet50 101 152卷积层模块
def conv_block(inputs, filter_num, stride=1, name=None):
    x = inputs
    x = Conv1D(filter_num[0], 5, strides=stride, padding='same', name=name + '_conv1',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(axis=-1, name=name + '_bn1')(x)
    x = Activation('relu', name=name + '_relu1')(x)

    x = Conv1D(filter_num[1], 5, strides=1, padding='same', name=name + '_conv2',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(axis=-1, name=name + '_bn2')(x)
    x = Activation('relu', name=name + '_relu2')(x)

    # x = Conv1D(filter_num[1], 3, strides=1, padding='same', name=name + '_conv3',
    #           kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization(axis=-1, name=name + '_bn3')(x)

    # residual connection
    r = Conv1D(filter_num[1], 1, strides=stride, padding='same', name=name + '_residual',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    x = layers.add([x, r])
    x = Activation('relu', name=name + '_relu3')(x)

    return x


def build_block(x, filter_num, blocks, stride=1, name=None):
    if blocks != 0:
        x = conv_block(x, filter_num, stride, name=name)
        for i in range(1, blocks):
            x = conv_block(x, filter_num, stride=1, name=name + '_block' + str(i))
    return x


# 创建resnet50 101 152
def resnet(net_name, nb_classes, input_length):
    resnet_config = {'ResNet50': [3, 4, 12, 0],
                     'ResNet101': [3, 4, 23, 3],
                     'ResNet152': [3, 8, 36, 3]}
    layers_dims = resnet_config[net_name]

    filter_block1 = [64, 64, 256]
    filter_block2 = [128, 128, 512]
    filter_block3 = [256, 256, 1024]
    filter_block4 = [512, 512, 2048]

    text_input = Input(shape=(input_length, 1))
    # stem block
    x = Conv1D(64, 5, strides=2, padding='same', name='stem_conv',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(text_input)
    x = BatchNormalization(axis=-1, name='stem_bn')(x)
    x = Activation('relu', name='stem_relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same', name='stem_pool')(x)
    # convolution block
    x = build_block(x, filter_block1, layers_dims[0], name='conv1')
    x = build_block(x, filter_block2, layers_dims[1], stride=2, name='conv2')
    x = build_block(x, filter_block3, layers_dims[2], stride=2, name='conv3')
    x = build_block(x, filter_block4, layers_dims[3], stride=2, name='conv4')
    # top layer
    x = GlobalAveragePooling1D(name='top_layer_pool')(x)
    x = Dropout(0.4)(x)
    x = Dense(nb_classes, activation='softmax', name='fc', kernel_initializer='he_normal')(x)

    model = models.Model(text_input, x, name=net_name)
    return model


def main():
    model = resnet('ResNet50', 1000, 1015)
    model.summary()


def cnn(num_classes):
    model = tf.keras.Sequential([
        Conv1D(64, 5, strides=1, padding='same', name='stem_conv',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), input_shape=(1015, 1)),
        MaxPooling1D(3, strides=2, padding='same'),
        Conv1D(64, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        Conv1D(64, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        MaxPooling1D(3, strides=2, padding='same'),
        Conv1D(128, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        Conv1D(128, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        MaxPooling1D(3, strides=2, padding='same'),
        Conv1D(256, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        Conv1D(256, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        MaxPooling1D(3, strides=2, padding='same'),
        Conv1D(512, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        Conv1D(512, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        GlobalAveragePooling1D(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')])
    return model


if __name__ == '__main__':
    main()
