from model import *
from dataset import *
import tensorflow as tf
import datetime
import tqdm
import os

learning_rate = 1e-3
epoch_num = 100
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class Train:
    def __init__(self, data, model_path, lr, epoch_num, len_dataset):

        if data == 'DNA':
            num_classes = 3
            data_length = 2089
        elif data == 'covid':
            num_classes = 2
            data_length = 900
        elif data == 'cell':
            1
        train_log_dir = 'logs/gradient_tape/' + data + '/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + data + '/' + current_time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.epoch = 0
        self.epoch_num = epoch_num
        self.train_FLAG = True
        self.model = resnet('ResNet50', num_classes, data_length)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        lr_sch = LinearDecay(lr, epoch_num * len_dataset, epoch_num * len_dataset / 2)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sch)
        self.train_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.test_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

        save_dir = model_path
        model_name = "weights-improvement.h5"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.filepath = os.path.join(save_dir, model_name)
        try:
            self.model.load_weights(self.filepath)
        except Exception as e:
            print(e)

    @tf.function
    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            predictions = self.model(x_train, training=True)
            loss = self.loss_object(y_train, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(y_train, predictions)

    @tf.function
    def test_step(self, x_test, y_test):
        predictions = self.model(x_test)
        loss = self.loss_object(y_test, predictions)

        self.test_loss(loss)
        self.test_accuracy(y_test, predictions)

    def __call__(self, train_dataset, val_dataset, test_dataset):
        val_acc = 0
        self.return_epoch()
        for epoch in range(self.epoch_num):
            if not self.train_FLAG:
                self.save()
                return 0
            self.epoch = epoch
            with tqdm.tqdm(train_dataset, total=len(train_dataset), ascii=True, position=0, leave=True) as t:
                for (x_train, y_train) in t:
                    t.set_description(f'Epoch Loop: {epoch + 1} Inner Epoch Loop:')
                    self.train_step(x_train, y_train)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)
            for (x_test, y_test) in val_dataset:
                self.test_step(x_test, y_test)
            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)

            template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result(),
                                  self.test_loss.result(),
                                  self.test_accuracy.result()))
            if self.test_accuracy.result() > val_acc:
                val_acc = self.test_accuracy.result()
                self.model.save(self.filepath)
            # Reset metrics every epoch
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
        for (x_test, y_test) in test_dataset:
            self.test_step(x_test, y_test)
        template = 'Test Loss: {}, Test Accuracy: {}'
        print(template.format(self.test_loss.result(), self.test_accuracy.result() * 100))

    def return_epoch(self):
        return self.epoch

    def save(self):
        self.model.save(self.filepath)
        print('Train Stop!')


class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate *
                            (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate


if __name__ == '__main__':
    data = 'covid'
    train_dataset, val_dataset, test_dataset, len_dataset = dataset(data)
    model_path = os.path.join('D:/DL/UiProject/saved_models', data)
    train = Train(data, model_path, 0.0001, epoch_num, len_dataset)
    train(train_dataset, val_dataset, test_dataset)
