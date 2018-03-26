import tensorflow as tf 
from tqdm import tqdm
from load_audio import next_verif_batch, load_full_dataset, load_from_file
import sklearn.metrics as sm
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
np.seterr(divide='ignore', invalid='ignore')


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.zeros(shape))

weights = {
        'wconv1':init_weights([3, 3, 1, 32]),
        'wconv2':init_weights([3, 3, 32, 128]),
        'wconv3':init_weights([3, 3, 128, 128]),
        'wconv4':init_weights([3, 3, 128, 192]),
        'wconv5':init_weights([3, 3, 192, 256]),
        'bconv1':init_biases([32]),
        'bconv2':init_biases([128]),
        'bconv3':init_biases([128]),
        'bconv4':init_biases([192]),
        'bconv5':init_biases([256]),
        'woutput':init_weights([256, 128]),
        'boutput':init_biases([128]),
        'woutput2':init_weights([256, 256]),
        'boutput2':init_biases([256]),
        'wfinal':init_weights([512, 2]),
        'bfinal':init_biases([2])}

def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def dual_cnn(X_first,X_second, weights, phase_train,keep_prob):
    
    
    x_first = X_first
    x_second = X_second
    #x_first = batch_norm(x_first, MEL_VALS, phase_train)
    #x_second = batch_norm(x_first, MEL_VALS, phase_train)

    x_first = tf.reshape(x_first,(-1,N_MELS,MEL_VALS,1))
    x_second = tf.reshape(x_second,(-1,N_MELS,MEL_VALS,1))
    print(x_first.shape)

    
    conv2_1_first = tf.add(tf.nn.conv2d(x_first, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
    conv2_1_second = tf.add(tf.nn.conv2d(x_second, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
    
    print(conv2_1_first.shape)
    conv2_1_first = tf.nn.relu(batch_norm(conv2_1_first, 32, phase_train))
    conv2_1_second = tf.nn.relu(batch_norm(conv2_1_second, 32, phase_train))
    
    mpool_1_first = tf.nn.max_pool(conv2_1_first, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    mpool_1_second = tf.nn.max_pool(conv2_1_second, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')

    dropout_1_first = tf.nn.dropout(mpool_1_first, keep_prob)
    dropout_1_second = tf.nn.dropout(mpool_1_second, keep_prob)

    conv2_2_first = tf.add(tf.nn.conv2d(dropout_1_first, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
    conv2_2_second = tf.add(tf.nn.conv2d(dropout_1_second, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
    
    print(conv2_2_first.shape)
    conv2_2_first = tf.nn.relu(batch_norm(conv2_2_first, 128, phase_train))
    conv2_2_second = tf.nn.relu(batch_norm(conv2_2_first, 128, phase_train))
    

    mpool_2_first = tf.nn.max_pool(conv2_2_first, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    mpool_2_second = tf.nn.max_pool(conv2_2_second, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    
    
    dropout_2_first = tf.nn.dropout(mpool_2_first, keep_prob)
    dropout_2_second = tf.nn.dropout(mpool_2_second, keep_prob)

    conv2_3_first = tf.add(tf.nn.conv2d(dropout_2_first, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
    conv2_3_second = tf.add(tf.nn.conv2d(dropout_2_second, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
    
    print(conv2_3_first.shape)
    conv2_3_first = tf.nn.relu(batch_norm(conv2_3_first, 128, phase_train))
    conv2_3_second = tf.nn.relu(batch_norm(conv2_3_second, 128, phase_train))
    
    mpool_3_first = tf.nn.max_pool(conv2_3_first, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    mpool_3_second = tf.nn.max_pool(conv2_3_second, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    

    dropout_3_first = tf.nn.dropout(mpool_3_first, keep_prob)
    dropout_3_second = tf.nn.dropout(mpool_3_second, keep_prob)

    conv2_4_first = tf.add(tf.nn.conv2d(dropout_3_first, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
    conv2_4_second = tf.add(tf.nn.conv2d(dropout_3_second, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
    print(conv2_4_first.shape)

    conv2_4_first = tf.nn.relu(batch_norm(conv2_4_first, 192, phase_train))
    conv2_4_second = tf.nn.relu(batch_norm(conv2_4_second, 192, phase_train))
    
    mpool_4_first = tf.nn.max_pool(conv2_4_first, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
    mpool_4_second = tf.nn.max_pool(conv2_4_second, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
    
    dropout_4_first = tf.nn.dropout(mpool_4_first, keep_prob)
    dropout_4_second = tf.nn.dropout(mpool_4_second, keep_prob)

    
    conv2_5_first = tf.add(tf.nn.conv2d(dropout_4_first, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
    conv2_5_second = tf.add(tf.nn.conv2d(dropout_4_second, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
    print(conv2_5_first.shape)
    
    conv2_5_first = tf.nn.relu(batch_norm(conv2_5_first, 256, phase_train))
    conv2_5_second = tf.nn.relu(batch_norm(conv2_5_second, 256, phase_train))
    
    mpool_5_first = tf.nn.max_pool(conv2_5_first, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='VALID')
    mpool_5_second = tf.nn.max_pool(conv2_5_second, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='VALID')
    
    dropout_5_first = tf.nn.dropout(mpool_5_first, keep_prob)
    dropout_5_second = tf.nn.dropout(mpool_5_second, keep_prob)
    print(dropout_5_second.shape)


    #flat_first = tf.reshape(dropout_4_first, [-1, weights['woutput'].get_shape().as_list()[0]])
    #flat_second = tf.reshape(dropout_4_second, [-1, weights['woutput'].get_shape().as_list()[0]])
    print(conv2_5_second.shape[1]*conv2_5_second.shape[2]*conv2_5_second.shape[3])
    flat_first = tf.reshape(dropout_5_first, [-1, dropout_5_first.shape[1]*dropout_5_first.shape[2]*dropout_5_first.shape[3]])
    flat_second = tf.reshape(dropout_5_second, [-1, dropout_5_second.shape[1]*dropout_5_second.shape[2]*dropout_5_second.shape[3]])

    print(flat_first.get_shape())
    
    flat_first = tf.add(tf.matmul(flat_first, weights['woutput2']), weights['boutput2'])
    flat_second = tf.add(tf.matmul(flat_second, weights['woutput2']), weights['boutput2'])
    
    flat_first = tf.nn.relu(flat_first)
    flat_second = tf.nn.relu(flat_second)
    # Apply Dropout
    flat_first = tf.nn.dropout(flat_first, keep_prob)
    flat_second = tf.nn.dropout(flat_second, keep_prob)
    
    print(flat_first.get_shape())
    print(flat_second.get_shape())
    
    final_layer = tf.concat([flat_first, flat_second],1)
    
    print(final_layer.get_shape())
    
    p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(final_layer,weights['wfinal']),weights['bfinal']))
    print(p_y_X.get_shape())
    
    return p_y_X




if __name__ == "__main__":

    #Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_type", help="Train or test",type = str, default='test')
    parser.add_argument("--batch_size", help="Batch Size",type = int,default = 25)
    parser.add_argument("--n_epoch", help="Number of epochs",type = int,default = 10)
    parser.add_argument("--keep_prob", help="Dropout probability to keep.",type = float,default = 0.5)

    #Only applicable during testing phase. IDs of audio files to be tested.
    parser.add_argument("--X1id", help="ID of the first audio file.",type = str,default = '2')
    parser.add_argument("--X2id", help="ID of the second audio file.",type = str,default = '5') 
    

    args = parser.parse_args()
    
    N_MELS = 96
    MEL_VALS = 938
    BATCH_SIZE = args.batch_size
    N_EPOCH = args.n_epoch
    KEEP_PROB = args.keep_prob
    X1testID = args.X1id
    X2testID = args.X2id
    PHASE_TYPE = args.phase_type
    PREFIX = 'melspect'
    checkpoint_folder = "./data/models/"
    mel_folder = "./data/melfeatures/"

    #Check for PHASE_TYPE and test the model on given IDs
    if PHASE_TYPE == 'test':
        #Check if checkpoint exists and then load
        if os.path.exists(os.path.join(checkpoint_folder,"model.ckpt.meta")):
            #Load the model
            sess=tf.Session()    
            #First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph(os.path.join(checkpoint_folder,'model.ckpt.meta'))
            saver.restore(sess,tf.train.latest_checkpoint(checkpoint_folder))

            # Now, let's access and create placeholders variables and
            # create feed-dict to feed new data
            graph = tf.get_default_graph()
            #Load the datapoints
            print(type(X1testID), type(X2testID))
            X1_file = [f for f in list(os.walk(mel_folder))[0][2] if f.split('_')[1] == X1testID][0]
            X2_file = [f for f in list(os.walk(mel_folder))[0][2] if f.split('_')[1] == X2testID][0]
            X1_label = X1_file.split('_')[2].split('.')[0]
            X2_label = X2_file.split('_')[2].split('.')[0]

            X1_features = load_from_file(os.path.join(mel_folder,X1_file))
            X2_features = load_from_file(os.path.join(mel_folder,X2_file))

            #Min max normalization
            X1_features = (X1_features - np.amin(X1_features))/(np.amax(X1_features) - np.amin(X1_features))
            X2_features = (X2_features - np.amin(X2_features))/(np.amax(X2_features) - np.amin(X2_features))
            if X1_label == X2_label: y_test = [1, 0]
            else: y_test = [0, 1]
            X1_features = np.array([X1_features])
            X2_features = np.array([X2_features])
            y_test = np.array([y_test])
            print("loaded")
            X_first = graph.get_tensor_by_name("X_first:0")
            X_second = graph.get_tensor_by_name("X_second:0")
            y = graph.get_tensor_by_name("y:0")
            phase_train = graph.get_tensor_by_name("phase_train:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            feed_dict = {X_first: X1_features,
                         X_second: X2_features,
                         y: y_test,
                         phase_train:False,
                         keep_prob: 1.0}
            #Now, access the op that you want to run. 
            op_to_restore = graph.get_tensor_by_name("predict_op:0")

            print(sess.run(op_to_restore,feed_dict))
            print("Whats up")
    else:  #If we want to train the model
        print("Training model")
        print("Done")
        print("Initializing graph.")
        
        X_first = tf.placeholder("float", [None, N_MELS, MEL_VALS, 1],name="X_first")
        X_second = tf.placeholder("float", [None, N_MELS, MEL_VALS, 1],name="X_second")

        y = tf.placeholder("float", [None, 2],name="y")
        lrate = tf.placeholder("float",name="lrate")
        keep_prob = tf.placeholder("float",name="keep_prob")
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        y_ = dual_cnn(X_first, X_second, weights, phase_train,keep_prob)
        
        predict_op = y_
        print(y.shape,y_.shape)
        # Train and Evaluate Model
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = y_))
        train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        print("Running session.")

        ##Initialize saver
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            #tf.initialize_all_variables().run()
            tf.global_variables_initializer().run()
            costs = []
            for i in tqdm(range(N_EPOCH)):
                batch_gen_train = next_verif_batch(BATCH_SIZE, type = 'mel',data='train', SAMPLE = 10016)
                count = 0
                costs_per_epoch = []
                while True:
                    try:
                        temp = next(batch_gen_train)
                        X1_train_batch,X2_train_batch,y_train_batch = temp[0], temp[1], temp[2]
                        
                        train_input_dict = {X_first: X1_train_batch,
                                            X_second: X2_train_batch,
                                            y: y_train_batch,
                                            phase_train: True,
                                            keep_prob: KEEP_PROB}
                        _, c = sess.run([train_op,cost], feed_dict=train_input_dict,options = run_options)
                        costs_per_epoch.append(c)

                        


                        if count%500== 0:
                            print("Cost after iteration %d in epoch %d is: %.5f"%(count, i, c))
                            
                            accuracies = []
                            F1_scores = []
                            batch_gen_test = next_verif_batch(BATCH_SIZE, type='mel',data='test',SAMPLE = 1216)

                            
                            while True:
                                try:
                                    
                                    temp_test = next(batch_gen_test)

                                    X1_test_batch,X2_test_batch,y_test_batch = temp_test[0], temp_test[1], temp_test[2]

                                    test_input_dict = {X_first: X1_test_batch,
                                                           X_second: X2_test_batch,
                                                           y: y_test_batch,
                                                           phase_train:False,
                                                           keep_prob: 1.0}
                                    predictions = sess.run(predict_op, feed_dict=test_input_dict).tolist()
                                    f1 = sm.f1_score(np.argmax(predictions,1).astype('float32'), 
                                                                        np.argmax(y_test_batch,1).astype('float32'))
                                    a = sm.accuracy_score(np.argmax(predictions,1).astype('float32'), 
                                                                        np.argmax(y_test_batch,1).astype('float32'))
                                    F1_scores.append(f1)
                                    accuracies.append(a)
                                    

                                    
                                except StopIteration:
                                    print('Epoch : ', i,  'Avg F1 score : ', np.mean(F1_scores), 
                                        'Avg. Accuracy: ', np.mean(accuracies))
                                    from sklearn.metrics import confusion_matrix
                                    confmat = confusion_matrix(np.argmax(y_test_batch,1).astype('float32'),
                                    	np.argmax(predictions,1).astype('float32'))  
                                    print(confmat)      
                                    break
                            

                        count += 1
                    except StopIteration:
                        #If the generator expires, then break out of the loop
                        costs.append(costs_per_epoch)
                        break
                


                print("We are in epoch: " +str(i))
                print("Cost:" +  str(c))
                train_accuracy = accuracy.eval(feed_dict=train_input_dict)
                print('step %d, training accuracy %g' % (i, train_accuracy))
                #Since we cannot calculate the accuracy of the whole test set at one time,
                #I decided to evaluate on batches of test data and then aggregate the results 
                #To calculate the overall accuracy.
                
                batch_gen_test = next_verif_batch(BATCH_SIZE, type='mel',data='test')
                while True:
                    try:
                        temp = next(batch_gen_test)
                        X1_test_batch,X2_test_batch,y_test_batch = temp[0], temp[1], temp[2]
                        test_input_dict = {X_first: X1_test_batch,
                                               X_second: X2_test_batch,
                                               y: y_test_batch,
                                               phase_train:False,
                                               keep_prob: 1.0}
                        predictions = sess.run(predict_op, feed_dict=test_input_dict).tolist()
                        F1_scores.append(sm.f1_score(np.argmax(predictions,1).astype('float32'), 
                                                            np.argmax(predictions,1).astype('float32')))
                        accuracies.append(accuracy.eval(feed_dict=test_input_dict))
                        
                    except StopIteration:
                        print('Epoch : ', i,  'Avg F1 score : ', np.mean(F1_scores), 'Avg. Accuracy: ', np.mean(accuracies))              
                        break
            """
            if not os.path.exists("./data/models/"):
                    os.mkdir("./data/models/")
            save_path = saver.save(sess, "./data/models/model.ckpt",global_step = N_EPOCH)
            print("Model saved in path: %s" % save_path)
            """
            plt.figure()

            import pickle
            pickle.dump(open("accuracies.pkl",'wb'),accuracies, protocol = 2)
            pickle.dump(open("F1_scores.pkl",'wb'),F1_scores, protocol = 2)
            pickle.dump(open("costs.pkl",'wb'),costs, protocol = 2)
            plt.plot(range(len(costs[0])),costs[0],'r')
            plt.plot(range(len(accuracies)), accuracies,'b')
