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
df = pd.read_csv("./augmented_image/metadata.csv")

# Path to store the tfrecord file
out_path_train = "./augmented_image/data_train.tfrecords"
out_path_valid = "./augmented_image/data_valid.tfrecords"
out_path_test = "./augmented_image/data_test.tfrecords"

image_paths = []
labels = []

for image, label in zip(df['image_id'], df['dx']):
    image_path = './augmented_image/' + image + '.jpg'
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

train_ = int(len(data) * 0.9)
test_ = int(len(data) - 100)

train = data.iloc[0:train_, :]
test = data.iloc[train_:test_, :]
valid = data.iloc[test_:len(data), :]

# Object of the dataset.
dataset_train = TfRecord(train['paths'], train['labels'], out_path_train)
dataset_valid = TfRecord(valid['paths'], valid['labels'], out_path_valid)
dataset_test = TfRecord(test['paths'], test['labels'], out_path_test)

# Converts and stores to the out_path.
dataset_train.convert_to_tfrecord()
dataset_valid.convert_to_tfrecord()
dataset_test.convert_to_tfrecord()



