from typing import Dict, Text
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow import keras
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


# Read the data
df = pd.read_csv('/home/harisyf/Documents/Bangkit 2022/Capstone Project/Dataset_Refield.csv')
# Take the necessary columns needed for training
rating_data = df[["User_ID","Field_ID", "User_Rating","Field_Name"]]
dataset = tf.data.Dataset.from_tensor_slices((tf.cast(rating_data['User_ID'].values.reshape(-1,1), tf.int32), 
                                              tf.cast(rating_data['Field_ID'].values.reshape(-1,1), tf.int32), 
                                              tf.cast(rating_data['User_Rating'].values.reshape(-1,1),tf.float32),
                                              tf.cast(rating_data['Field_Name'].values.reshape(-1,1), tf.string)))

#Change data type for the tf
field = rating_data.Field_ID.values
users = rating_data.User_ID.values
name = rating_data.Field_Name.values

unique_field_id = np.unique(list(field))
unique_user_ids = np.unique(list(users))
unique_field_name = np.unique(list(name))


#Rename the column
def rename(x0,x1,x2,x3):
    y = {}
    y["User_ID"] = x0
    y['Field_ID'] = x1
    y['User_Rating'] = x2
    y['Field_Name'] = x3
    return y

dataset = dataset.map(rename)

#Refield Model
class RefieldModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.IntegerLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for field.
    self.field_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_field_name, mask_token=None),
      tf.keras.layers.Embedding(len(unique_field_name) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Final layer.
      tf.keras.layers.Dense(1)
  ])

  def __call__(self, x):
    
    User_ID, Field_Name = x
    user_embedding = self.user_embeddings(User_ID)
    field_embedding = self.field_embeddings(Field_Name)

    return self.ratings(tf.concat([user_embedding, field_embedding], axis=1))

#define network task with tf recommender
class FieldModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RefieldModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def compute_loss(self, features, training=False) -> tf.Tensor:
    print(features)
    rating_predictions = self.ranking_model((features['User_ID'], features["Field_Name"]))

    # The task computes the loss and the metrics.
    return self.task(labels=features["User_Rating"], predictions=rating_predictions)

#split train and test data
tf.random.set_seed(42)
shuffled = dataset.shuffle(8902, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(7122)
test = shuffled.skip(7122).take(1780)


#compile and train the model
model = FieldModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.4))
history = model.fit(train, validation_data = test, epochs=15)


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#save model
refield_model= '/home/harisyf/Documents/Bangkit 2022/Capstone Project/refield-model'
tf.saved_model.save(model,export_dir=refield_model)

loaded = tf.saved_model.load(refield_model)

# Convert the saved model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(refield_model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
tflite_model = converter.convert()

tflite_model_file = 'converted_model.tflite'

with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)