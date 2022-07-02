import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import plot_model
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Image loading

rcParams['figure.figsize'] = 20, 10
data_path = './data/ck/CK+48'
data_dir_list = os.listdir(data_path)

num_epoch = 10

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (48, 48))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data / 255
img_data.shape

num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:134] = 0  # 135
labels[135:188] = 1  # 54
labels[189:365] = 2  # 177
labels[366:440] = 3  # 75
labels[441:647] = 4  # 207
labels[648:731] = 5  # 84
labels[732:980] = 6  # 249

names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

def get_label(label_id):
    return ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'][label_id]

# data preparing

Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)
X_train = X_train
x_test = X_test

# CNN layer creating

def create_CNN():
    input_shape = (48, 48, 3)
    model = Sequential([
        Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', input_shape=input_shape),
        MaxPooling2D(2),
        Conv2D(64, 3, activation='relu', kernel_initializer='he_normal'),
        MaxPooling2D(2),
        Dropout(0.25),
        Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', input_shape=input_shape),
        MaxPooling2D(2),
        Conv2D(256, 3, activation='relu', kernel_initializer='he_normal'),
        MaxPooling2D(2),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax', kernel_initializer='glorot_normal')

    ])
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='RMSprop')
    return model


model_custom = create_CNN()
model_custom.get_config()
model_custom.layers[0].get_config()
model_custom.layers[0].input_shape
model_custom.layers[0].output_shape
model_custom.layers[0].get_weights()
np.shape(model_custom.layers[0].get_weights()[0])
model_custom.layers[0].trainable
model_custom.summary()

plot_model(model_custom, to_file='model.png')

# Model testing

kf = KFold(n_splits=5, shuffle=False)

train_datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True,
    fill_mode="nearest")

BS = 8
EPOCHS = 200
result = []
scores_loss = []
scores_acc = []
k_no = 0
for train_index, test_index in kf.split(x):
    X_Train_ = x[train_index]
    Y_Train = y[train_index]
    X_Test_ = x[test_index]
    Y_Test = y[test_index]

    file_path = "/kaggle/working/weights_best_" + str(k_no) + ".hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="loss", mode="min", patience=8)

    callbacks = [checkpoint, early]

    #es = EarlyStopping(
    #    monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=2,
    #    mode='max', baseline=None, restore_best_weights=True
    #)
    #lr = ReduceLROnPlateau(
    #    monitor='val_accuracy', factor=0.1, patience=5, verbose=2,
    #    mode='max', min_delta=1e-5, cooldown=0, min_lr=0
    #)

    #callbacks = [es, lr]

    model = create_CNN()
    hist = model.fit(train_datagen.flow(X_Train_, Y_Train), epochs=EPOCHS,
                     validation_data=(X_Test_, Y_Test),
                     callbacks=callbacks, verbose=0)
    # model.load_weights(file_path)
    result.append(model.predict(X_Test_))
    score = model.evaluate(X_Test_, Y_Test, verbose=0)
    scores_loss.append(score[0])
    scores_acc.append(score[1])
    k_no += 1

# Printing scores

print(scores_acc, scores_loss)

value_min = min(scores_loss)
value_index = scores_loss.index(value_min)
print(value_index)

# model.load_weights("/kaggle/working/weights_best_" + str(value_index) + ".hdf5")
best_model = model

score = best_model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print(test_image.shape)

predict_x = model.predict(test_image)
classes_x = np.argmax(predict_x, axis=1)

print(predict_x)
print(classes_x)
print(y_test[0:1])

predict_x = model.predict(X_test[9:18])
res = np.argmax(predict_x, axis=1)
plt.figure(figsize=(10, 10))

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_test[i], cmap=plt.get_cmap('gray'))
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('prediction = %s' % get_label(res[i]), fontsize=14)
# show the plot
plt.show()

# predict
y_pred = best_model.predict(X_test)

# visualizing losses and accuracy

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

epochs = range(len(train_acc))

plt.plot(epochs, train_loss, 'r', label='train_loss')
plt.plot(epochs, val_loss, 'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, 'r', label='train_acc')
plt.plot(epochs, val_acc, 'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.legend()
plt.figure()

best_model.save_weights('model_weights.h5')
best_model.save('model_keras.h5')



predict_x = model.predict(X_test)
results = np.argmax(predict_x, axis=1)

cm = confusion_matrix(np.where(y_test == 1)[1], results)
# cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]

label_mapdisgust = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm, index=label_mapdisgust,
                     columns=label_mapdisgust
                     )
final_cm = cm_df
final_cm

plt.figure(figsize=(5, 5))
sns.heatmap(final_cm, annot=True, cmap='Greys', cbar=False, linewidth=2, fmt='d')
plt.title('CNN Emotion Classify')
plt.ylabel('True class')
plt.xlabel('Prediction class')
plt.show()

from sklearn.metrics import roc_curve, auc
from itertools import cycle

new_label = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
final_label = new_label
new_class = 7
# ravel flatten the array into single vector
y_pred_ravel = y_pred.ravel()
lw = 2
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(new_class):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# colors = cycle(['red', 'green','black'])
colors = cycle(['red', 'green', 'black', 'blue', 'yellow', 'purple', 'orange'])
for i, color in zip(range(new_class), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0}'''.format(final_label[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
