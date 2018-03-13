import cv2
import numpy as np
import glob
import sys
from sklearn.model_selection import train_test_split

print('Loading training data...')
e0 = cv2.getTickCount()

# load training data
image_array = np.zeros((1, 38400), dtype=np.float32)
label_array = np.zeros((1, 4), dtype=np.float32)
training_data = glob.glob('training_data/*.npz')


# if no data, exit
if not training_data:
    print("No training data in directory, exit")
    sys.exit()

for single_npz in training_data:
    with np.load(single_npz) as data:
        train_temp = data['train']
        train_labels_temp = data['train_labels']
    print(train_temp)
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

X = image_array[1:, :].astype('float32')
y = label_array[1:, :].astype('float32')


print('Image array shape: {}'.format(X.shape))
print('Label array shape: {}'.format(y.shape))

e00 = cv2.getTickCount()
time0 = (e00 - e0)/ cv2.getTickFrequency()
print('Loading image duration: {}'.format(time0))

# train test split, 7:3
train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.1)


# set start time
e1 = cv2.getTickCount()

# create MLP
layer_sizes = np.array([38400, 32, 4], dtype=np.uint16)

model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(layer_sizes)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

num_iter = model.train(train, cv2.ml.ROW_SAMPLE, train_labels)
# Can't accept float64 training data

# set end time
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print('Training duration: {}'.format(time))
#print 'Ran for %d iterations' % num_iter

# train data
ret_0, resp_0 = model.predict(train)
prediction_0 = resp_0.argmax(-1)
true_labels_0 = train_labels.argmax(-1)

train_rate = np.mean(prediction_0 == true_labels_0)
print('Train accuracy: ', "{0:.2f}%".format(train_rate * 100))

# test data
ret_1, resp_1 = model.predict(test)
prediction_1 = resp_1.argmax(-1)
true_labels_1 = test_labels.argmax(-1)

test_rate = np.mean(prediction_1 == true_labels_1)
print('Test accuracy: ', "{0:.2f}%".format(test_rate * 100))

# save model
model.save('mlp_xml/mlp.xml')
