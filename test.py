from tf_records import DataLoad
import tensorflow as tf
import tensorflow_hub as hub
from tensorboard.plugins.beholder import Beholder

LOGDIR = 'two'
BATCH_SIZE = 32
EPOCHS = 2
LR = 1e-5

dataset_train = DataLoad('./HAM10000/data_train.tfrecords', EPOCHS, BATCH_SIZE).return_dataset()
dataset_test = DataLoad('./HAM10000/data_train.tfrecords', 1, 1000).return_dataset()

with tf.Graph().as_default():

    def network(x):
        module_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
        mod = hub.Module(module_url)
        x_image = tf.reshape(x, [-1, 299, 299, 3])
        features = mod(x_image)   
        
        tf.summary.image('input', x_image, 3)
        fc = tf.layers.dense(inputs=features, units=200, activation=tf.nn.relu)
        layer = tf.layers.dense(inputs=fc, units=7, activation=None)
        return layer

    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, shape=[None, 299 * 299 * 3], name="x")

        y = tf.placeholder(tf.float32, shape=[None, 7], name="labels")
 
    logits = network(x)
            
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(LR, name='Adam').minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("accuracy_val"):
        correct_prediction_val = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy_val = tf.reduce_mean(tf.cast(correct_prediction_val, tf.float32))
        tf.summary.scalar("accuracy_val", accuracy_val)

    summ = tf.summary.merge_all()

    saver = tf.train.Saver()

    iterator = dataset_train.make_one_shot_iterator()
    next_element = iterator.get_next() 

    # iterator_test = dataset_test.make_one_shot_iterator()
    # test = iterator_test.get_next() 
    with tf.Session() as sess:
            
        beholder = Beholder(LOGDIR)
        
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOGDIR)
        writer.add_graph(sess.graph)
        # test_value = sess.run(test)
        i = 0
        while(True):

            try:
                print(i)
                i += 1
                value = sess.run(next_element)
                x_ = value[0]
                y_ = value[1]
                if i % 310 == 0:
                    [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: x_, y: y_})
                    writer.add_summary(s, i)      
                    # [test_accuracy, s] = sess.run([accuracy_val, summ], feed_dict={x: test_value[0][0:100], y: test_value[1][0:100]})
                    # writer.add_summary(s, i)            
                sess.run(train_step, feed_dict={x: x_, y: y_})
                beholder.update(session=sess)

                export_path =  './savedmodel1/' + str(i)
                builder = tf.saved_model.builder.SavedModelBuilder(export_path)

                tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
                tensor_info_y = tf.saved_model.utils.build_tensor_info(logits)

                prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'x_input': tensor_info_x},
                    outputs={'y_output': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

                builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        prediction_signature 
                },
                )
                builder.save()
                if i == 3:
                    break
            except:
                break


   

with tf.Session() as sess:
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_key = 'x_input'
    output_key = 'y_output'

    export_path =  './savedmodel1/2'
    meta_graph_def = tf.saved_model.loader.load(
               sess,
              [tf.saved_model.tag_constants.SERVING],
              export_path)
    signature = meta_graph_def.signature_def

    x_tensor_name = signature[signature_key].inputs[input_key].name
    y_tensor_name = signature[signature_key].outputs[output_key].name

    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)

    y_out = sess.run(y, {x: [x_[0]]})
    print(y_out)