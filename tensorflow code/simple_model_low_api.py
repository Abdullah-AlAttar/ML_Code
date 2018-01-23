import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.keras as keras

x_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1,
                    2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0,
                    9.0])


class TfLinreg:
    def __init__(self, x_dim, lr=0.01, random_state=None):
        self.x_dim = x_dim
        self.lr = lr
        self.g = tf.Graph()

        with self.g.as_default():
            tf.set_random_seed(random_state)
            self.build()
            self.init_op = tf.global_variables_initializer()

    def build(self):
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=(None, self.x_dim),
                                name='x_input')
        self.y = tf.placeholder(dtype=tf.float32,
                                shape=(None),
                                name='y_input')
        print(self.X)
        print(self.y)
        w = tf.Variable(tf.zeros(shape=(1), name='weight'))
        b = tf.Variable(tf.zeros(shape=(1), name='bias'))
        print(w)
        print(b)
        self.z_net = tf.squeeze(w * self.X + b, name='z_net')
        print(self.z_net)
        sqr_errors = tf.square(self.y - self.z_net, name='sqr_errors')
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.lr,
            name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)


def train_linreg(sess, model, X_train, y_train, num_epochs=10):
    sess.run(model.init_op)
    costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost],
                           feed_dict={model.X: X_train,
                                      model.y: y_train})
        costs.append(cost)
    return costs


def predict_linreg(sess, model, X_test):
    y_pred = sess.run(model.z_net,
                      feed_dict={model.X: X_test})
    return y_pred


lrModel = TfLinreg(x_dim=x_train.shape[1], lr=0.01)

sess = tf.Session(graph=lrModel.g)

training_costs = train_linreg(sess, lrModel, x_train, y_train)
plt.plot(range(1, len(training_costs) + 1), training_costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Trainst cost')
plt.show()

plt.scatter(x_train, y_train,
            marker='s', s=50,
            label='Training Data')
plt.plot(range(x_train.shape[0]),
         predict_linreg(sess, lrModel, x_train),
         color='gray', marker='o',
         markersize=6, linewidth=3,
         label='LinReg Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()
