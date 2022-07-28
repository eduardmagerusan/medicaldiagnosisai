import os
import keras.utils
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, request, render_template, redirect, url_for
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
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
            # np.array(image.load_img(image_path, target_size=(H, W))))
            np.array(keras.utils.load_img(image_path, target_size=(h, w))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std


def load_image(img, image_dir, df, preprocess=True, h=320, w=320):
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, h=h, w=w)
    # x = image.load_img(img_path, target_size=(H, W))
    x = keras.utils.load_img(img_path, target_size=(h, w))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
image_dir = "/Users/eduardmagerusan/Downloads/images/"

image_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

train = image_generator.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="Image",
        y_col=labels,
        class_mode="raw",
        batch_size=8,
        shuffle=True,
        seed=1,
        target_size=(320, 320))

raw_train_generator = ImageDataGenerator().flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col="Image",
    y_col=labels,
    class_mode="raw",
    batch_size=100,
    shuffle=True,
    target_size=(320, 320))

batch = raw_train_generator.next()
data_sample = batch[0]

image_generator.fit(data_sample)

test = image_generator.flow_from_dataframe(
    dataframe=test_df,
    directory=image_dir,
    x_col="Image",
    y_col=labels,
    class_mode="raw",
    batch_size=8,
    shuffle=False,
    seed=1,
    target_size=(320, 320))

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

predicted_vals = model.predict_generator(test, steps=len(test))

pd.DataFrame(classification_report(test.labels, predicted_vals > 0.5, output_dict=True, target_names=labels))

auc_roc_vals = []
for i in range(len(labels)):
    gt = test.labels[:, i]
    pred = predicted_vals[:, i]
    auc_roc = roc_auc_score(gt, pred)
    auc_roc_vals.append(auc_roc)
    fpr_rf, tpr_rf, _ = roc_curve(gt, pred)

df = pd.read_csv("train.csv")

labels_to_show = np.take(labels, np.argsort(auc_roc_vals)[::-1])[:6]

# generate_gradcam(model, '00000091_002.png', image_dir, df, labels, labels_to_show)
# plt.show()

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    filename = secure_filename(imagefile.filename)
    imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    preprocessed_input = load_image(imagefile.filename, image_dir, df)

    with session.as_default():
        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(session)
            predictions = model.predict(preprocessed_input)

    label_list = []
    prediction_values = []
    for i in range(len(labels)):
        if labels[i] in labels_to_show:
            print(f"Generating gradcam for class {labels[i]}")
            label_list.append(f"{labels[i]}")
            # label = f"{labels[i]}"
            prediction_values.append(f"{predictions[0][i]*100:.1f}")
            # prediction_value = f"{predictions[0][i]*100:.1f}"

    return render_template('index.html', filename=filename, prediction1=prediction_values[0],
                           prediction2=prediction_values[1], prediction3=prediction_values[2],
                           prediction4=prediction_values[3], prediction5=prediction_values[4],
                           prediction6=prediction_values[5])


if __name__ == "__main__":
    app.run()
