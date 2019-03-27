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
out_path = "./HAM10000/data.tfrecords"

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

# Object of the dataset.
dataset = TfRecord(image_paths, labels, out_path)

# Converts and stores to the out_path.
dataset.convert_to_tfrecord()



