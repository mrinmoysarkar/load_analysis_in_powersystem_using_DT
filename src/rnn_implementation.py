import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, mean_squared_log_error, median_absolute_error, r2_score



def MASE(training_series, testing_series, prediction_series, m=365):
    n = training_series.shape[0]
    d = np.abs(  np.diff( training_series) ).sum()/(n-1)
#     i=0
#     d = 0
#     while (i+m) < n:
#         d += np.abs(training_series[i]-training_series[i+m])
#         i += 1
#     d = d/(n-m)
    errors = np.abs(testing_series - prediction_series )
    return errors.mean()/d

def MAPE(y_true, y_pred):
    mape = 100*sum(np.divide(np.abs(y_true-y_pred),y_true))/len(y_true)
    return mape[0]

def next_batch(training_data,batch_size,steps,start_point):
    # Grab a random starting point for each batch
    # rand_start = np.random.randint(0,len(training_data)-steps*2)
    rand_start = start_point
    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    X_batch = np.array(training_data[rand_start:rand_start+steps]).reshape(-1, steps, 1)
    y_batch = np.array(training_data[rand_start+steps:rand_start+steps*2]).reshape(-1, steps, 1)
    start_point += steps
    if (start_point + 2*steps) >= len(training_data):
        start_point = 0
    return X_batch, y_batch, start_point



if __name__=="__main__":
    # datasetnames = ["coast", "east", "ercot", "far_west", "north", "north_c", "south_c", "southern", "west"]

    datasetnames=["east"]
    datasetNamem = "coast"

    for dtindx in range(len(datasetnames)):

        datasetName = datasetnames[dtindx]
        train = pd.read_csv('../data/Daily_Load_Data/train/' + datasetName + '.csv', index_col='Time')
        test = pd.read_csv('../data/Daily_Load_Data/test/' + datasetName + '.csv', index_col='Time')
        train = train.dropna(axis=0)
        test = test.dropna(axis=0)

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train)

        # Just one feature, the time series
        num_inputs = 1
        # Num of steps in each batch
        num_time_steps = 365
        # 100 neuron layer, play with this
        num_neurons = 1500
        # number of layer
        num_layers = 3
        # Just one output, predicted time series
        num_outputs = 1
        ## You can also try increasing iterations, but decreasing learning rate
        # learning rate you can play with this
        learning_rate = 0.001
        # how many iterations to go through (training steps), you can play with this
        num_train_iterations = 2000
        # Size of the batch of data
        batch_size = 1

        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
        y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

        cell = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.GRUCell(num_units=num_neurons, activation=tf.nn.relu)
             for layer in range(num_layers)]),
            output_size=num_outputs)
        #     cell = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.MultiRNNCell(
        #         [tf.nn.rnn_cell.LSTMCell(num_units=num_neurons, activation=tf.nn.relu)
        #                                         for layer in range(num_layers)]),
        #                                         output_size=num_outputs)
        outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        loss = tf.reduce_mean(tf.square(outputs - y))  # MSE
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss)
        init = tf.global_variables_initializer()

        model_file_name = "../../temp/model_" + datasetNamem + ".ckpt"
        saver = tf.train.Saver()
        start_point = 0
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            # sess.run(init)
            saver.restore(sess, model_file_name)
            for iteration in range(num_train_iterations + 1):
                X_batch, y_batch, start_point = next_batch(train_scaled, batch_size, num_time_steps, start_point)
                sess.run(train, feed_dict={X: X_batch, y: y_batch})
                if iteration % 1000 == 0 and iteration != 0:
                    try:
                        # mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                        # print(iteration, "\tMSE:", mse)

                        # calculate train error
                        s_point = 0
                        mse = 0
                        mae = 0
                        mase = 0
                        mape = 0
                        trun = 10
                        for i in range(trun):
                            X_batch, y_batch, s_point = next_batch(train_scaled, batch_size, num_time_steps, s_point)
                            y_pred = sess.run(outputs, feed_dict={X: X_batch})
                            results = scaler.inverse_transform(y_pred.reshape(num_time_steps, 1))
                            true_data = y_batch.reshape(num_time_steps, 1)
                            true_data = scaler.inverse_transform(true_data)
                            n = min(results.shape[0], true_data.shape[0])

                            y_pred = results[:n]
                            y_true = true_data[:n]

                            mse += mean_squared_error(y_true=y_true, y_pred=y_pred)
                            mae += mean_absolute_error(y_true=y_true, y_pred=y_pred)

                            train_series = scaler.inverse_transform(train_scaled)
                            mase += MASE(train_series.ravel(), y_true.ravel(), y_pred.ravel())
                            mape += MAPE(y_true, y_pred)
                        mse /= trun
                        mae /= trun
                        mase /= trun
                        mape /= trun
                        print(3000 + iteration, " train errors:\t" + datasetName + '\tRMSE:', round(mse ** 0.5, 4),
                              '\tMAE:',
                              round(mae, 4), '\tMASE:', round(mase, 4), '\tMAPE:', round(mape, 4))

                        # test error
                        train_seed = list(train_scaled[-num_time_steps:])
                        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
                        y_pred = sess.run(outputs, feed_dict={X: X_batch})
                        results = scaler.inverse_transform(y_pred.reshape(num_time_steps, 1))
                        true_data = test.values
                        n = min(results.shape[0], true_data.shape[0])

                        y_pred = results[:n]
                        y_true = true_data[:n]

                        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
                        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)

                        train_series = scaler.inverse_transform(train_scaled)
                        mase = MASE(train_series.ravel(), y_true.ravel(), y_pred.ravel())
                        mape = MAPE(y_true, y_pred)

                        print(3000 + iteration, " test  errors:\t" + datasetName + '\tRMSE:', round(mse ** 0.5, 4),
                              '\tMAE:',
                              round(mae, 4), '\tMASE:', round(mase, 4), '\tMAPE:', round(mape, 4))
                        saver.save(sess, model_file_name)
                    except Exception as e:
                        print("error1")
                        break