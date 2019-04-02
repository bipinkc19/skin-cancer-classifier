# Skin Cancer Classifier

Convolutional Neural Network for classifying canceorus moles from images.

# Setting up locally
```bash
$ git clone git@gitlab.lftechnology.com:leapfrogai/skin-cancer-detection.git
```

## Create a virtual environment.

```bash
$ python -m venv venv
```

Make sure you are using `python3.6`.

## Activate the virtual environment.

```bash
$ source venv/bin/activate
```

## Install the dependencies.

```bash
$ pip install -r requirements.txt
```

## Download HAM10000 dataset.
```bash
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Extract the two zip and the metadata csv to HAM10000 folder.
```

## Prepare the dataset.
```bash
$ python create_dataset.py
```

## To train the model.
```bash
$ python train.py
```

## For tensorboard vizualization while training.
```bash
$ tensorboard --logdir=tensorboard_logs --host=localhost --port=8088
```
## For testing after completion.
```bash
$ python test.py
```

## Saved models.

All models are save each time after 10 epochs in saved_model. Each model can be tested by extracting by setting the import_path in `test.py`.

## Note:

Different model require different size if images. This can be edited in the `tf_records.py` in resize opereation.

