from tf_records import DataLoad
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns  
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

def classification_metrics(actual, pred, msg):
    cm = confusion_matrix(actual, pred)
       
    plt.figure()
    ax= plt.subplot()
    sns.heatmap(cm, annot = True, fmt = 'g')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6']) 
    ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6'])   
    plt.show()       
    
    print(classification_report(actual, pred))

def array_argmax(value):
    args = []
    for i in range(len(value)):
        index_max = np.argmax(value[i])
        args.append(index_max)
    
    return args
    
dataset_test = DataLoad('./HAM10000/data_test.tfrecords', 1, 1, 32).return_dataset()

iterator_test = dataset_test.make_one_shot_iterator()
test = iterator_test.get_next()

i = 0
with tf.Session() as sess:

    import_path = "./savedmodel/batch_64_lr_1e_resnetv2_100/..."
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_key = 'x_input'
    output_key = 'y_output'
    meta_graph_def = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            import_path)
    signature = meta_graph_def.signature_def
    x_tensor_name = signature[signature_key].inputs[input_key].name
    y_tensor_name = signature[signature_key].outputs[output_key].name
    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)
    while(True):
        try:
            value_ = sess.run(test)
            predicted_ = sess.run(y, {x: value_[0]})
            actual_ = value_[1]
            if i == 0:
                actual = actual_
                predicted = predicted_
            else:
                actual = np.concatenate((actual, actual_))
                predicted = np.concatenate((predicted, predicted_))
            i = 1
        except tf.errors.OutOfRangeError:
            break

classification_metrics(array_argmax(actual), array_argmax(predicted), msg = '')