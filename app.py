import os
import keras.utils
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, request, render_template, redirect, url_for
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

from werkzeug.utils import secure_filename

import random


tf.compat.v1.disable_eager_execution()

app = Flask(__name__)

labels = ['Cardiomegaly',
          'Emphysema',
          'Effusion',
          'Hernia',
          'Infiltration',
          'Mass',
          'Nodule',
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening',
          'Pneumonia',
          'Fibrosis',
          'Edema',
          'Consolidation']


random.seed(a=None, version=2)


def get_mean_std_per_batch(image_path, df, h=320, w=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        # path = image_dir + img
        sample_data.append(
            np.array(keras.utils.load_img(image_path, target_size=(h, w))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std


def load_image(img, image_dir, df, preprocess=True, h=320, w=320):
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, h=h, w=w)
    x = keras.utils.load_img(img_path, target_size=(h, w))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
image_directory = 'static/uploads/'


base_model = DenseNet121(weights='./models/densenet.hdf5',
                         include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(labels), activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.call = tf.function(model.call)


config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)
model.load_weights("./models/pretrained_model.h5")

graph = tf.compat.v1.get_default_graph()

dataframe = pd.read_csv("train.csv")

labels_to_show = ['Cardiomegaly', 'Edema', 'Mass', 'Emphysema', 'Pneumothorax', 'Atelectasis']


UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    filename = secure_filename(imagefile.filename)
    imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    preprocessed_input = load_image(imagefile.filename, image_directory, dataframe)

    with session.as_default():
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(session)
            predictions = model.predict(preprocessed_input)

    label_list = []
    prediction_values = []
    for i in range(len(labels)):
        if labels[i] in labels_to_show:
            label_list.append(f"{labels[i]}")
            prediction_values.append(f"{predictions[0][i]*100:.1f}")

    return render_template('index.html', filename=filename, prediction1=prediction_values[0],
                           prediction2=prediction_values[1], prediction3=prediction_values[2],
                           prediction4=prediction_values[3], prediction5=prediction_values[4],
                           prediction6=prediction_values[5])


if __name__ == "__main__":
    app.run()
