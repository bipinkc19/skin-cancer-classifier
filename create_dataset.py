import pandas as pd
from tf_records import TfRecord

def label_to_array(label):
    ''' One hot encode the labels. '''

    if label == 'akiec':
        
        return([1, 0, 0, 0, 0, 0, 0])
    elif label == 'bcc':
        
        return([0, 1, 0, 0, 0, 0, 0])
    elif label == 'bkl':
        
        return([0, 0, 1, 0, 0, 0, 0])
    elif label == 'df':
        
        return([0, 0, 0, 1, 0, 0, 0])
    elif label == 'mel':
        
        return([0, 0, 0, 0, 1, 0, 0])
    elif label == 'nv':
        
        return([0, 0, 0, 0, 0, 1, 0])
    elif label == 'vasc':
        
        return([0, 0, 0, 0, 0, 0, 1])
    else:
        
        return([0, 0, 0, 0, 0, 0, 0])

# Read the metadata for images and their corresponding labels.
df = pd.read_csv("./HAM10000/HAM10000_metadata.csv")

# Path to store the tfrecord file
out_path_train = "./HAM10000/data_train.tfrecords"
out_path_test = "./HAM10000/data_test.tfrecords"

image_paths = []
labels = []

for image, label in zip(df['image_id'], df['dx']):
    image_path = './HAM10000/' + image + '.jpg'
    label_num = label_to_array(label)

    # Exceptions where the label doesn't matches any of the labels are discarded.
    if sum(label_num) == 0:
        continue
        
    image_paths.append(image_path)
    labels.append(label_num)

data = pd.DataFrame()
data['paths'] = image_paths
data['labels'] = labels

data = data.sample(frac=1)

train = data.iloc[0:9500, :]
test = data.iloc[9500:10014, :]
# Object of the dataset.
dataset_train = TfRecord(train['paths'], train['labels'], out_path_train)
dataset_test = TfRecord(test['paths'], test['labels'], out_path_test)

# Converts and stores to the out_path.
dataset_train.convert_to_tfrecord()
dataset_test.convert_to_tfrecord()



