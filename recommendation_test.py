from capstone_refield import *
import numpy as np
import tensorflow as tf

model = tf.saved_model.load('/home/harisyf/Documents/Bangkit 2022/Capstone Project/refield-model')

#Recommendation Testing
# Create array with users id in every place on the length of the field
#input the user id
id_ = input('Enter the user_id: ')
user = np.array([id_ for i in range(len(unique_field_id))])
vector = np.vectorize(np.int32)
user = vector(user)
# Convert it to tf.data.Dataset 
test_data = tf.data.Dataset.from_tensor_slices((tf.cast(user.reshape(-1,1), tf.int32), 
                                                tf.cast(unique_field_name.reshape(-1,1), tf.string)))
# rename the columns 
def rename_test(x0,x1):
    y = {}
    y["User_ID"] = x0
    y['Field_Name'] = x1
    return y
test_data = test_data.map(rename_test)

# test the predictions and store them in to dictionary
test_ratings = {}
for f in test_data:
    test_ratings[f['Field_Name'].numpy()[0]] = model.ranking_model((f['User_ID'],f['Field_Name']))

# sort them by score and print the field name
for f in sorted(test_ratings, key=test_ratings.get):
    print(f.decode('utf-8'))