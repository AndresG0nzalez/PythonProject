#Code was used from https://github.com/jeremyjordan/machine-learning/blob/master/projects/dog-project/dog_app.ipynb

#Need to download Bottleneck feature from https://www.floydhub.com/s_joy/datasets/bottleneck_features/3/DogInceptionV3Data.npz




from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tqdm import tqdm
import numpy as np
from glob import glob
import cv2
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
random.seed(8675309)


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
dog_breeds = len(dog_names)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

img_width, img_height = 224, 224

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_width, img_height))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))




#
#Detecting Human Faces
#
# extract pre-trained face detector
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#
# # load color (BGR) image
# img = cv2.imread(human_files[1])
# # convert BGR image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # find faces in image
# faces = face_cascade.detectMultiScale(gray)
#
# # print number of faces detected in the image
# print('Number of faces detected:', len(faces))
#
# # get bounding box for each detected face
# for (x, y, w, h) in faces:
#     # add bounding box to color image
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
# # convert BGR image to RGB for plotting
# cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # display the image, along with bounding box
# plt.imshow(cv_rgb)
# plt.show()
#
#
#Detecting Human Faces
#






#
#Assessing the Human Face Detector
#
# Select subset of data for faster evaluation
# human_files_short = human_files[:100]
# dog_files_short = train_files[:100]
#
# # Vectorize the face dectector function
# faces_vfunc = np.vectorize(face_detector)
#
# # Detect faces in both sets
# human_faces = faces_vfunc(human_files_short)
# dog_faces = faces_vfunc(dog_files_short)
#
# # Calculate and print percentage of faces in each set
# print('Faces detected in {:.2f}% of the sample human dataset.'.format((sum(human_faces)/len(human_faces))*100))
# print('Faces detected in {:.2f}% of the sample dog dataset.'.format((sum(dog_faces)/len(dog_faces))*100))
# #
#Assessing the Human Face Detector
#


#
#Assessing the Dog Detector
#
# Files already loaded in previous cell

# Vectorize the face dectector function
# dog_vfunc = np.vectorize(dog_detector)

# Detect dogs in both sets
# human_dogs = dog_vfunc(human_files_short)
# dog_dogs = dog_vfunc(dog_files_short)

# Calculate and print percentage of faces in each set
# print('Dogs detected in {:.2f}% of the sample human dataset.'.format((sum(human_dogs)/len(human_dogs))*100))
# print('Dogs detected in {:.2f}% of the sample dog dataset.'.format((sum(dog_dogs)/len(dog_dogs))*100))
#
#
#Assessing the Dog Detector
#


#
#
#Obtain Bottleneck Features
#
#
bottleneck_features = np.load('DogInceptionV3Data.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']
train_InceptionV3.shape[1:]
#
#
#Obtain Bottleneck Features
#
#



#
#
#Model Architecture
#
#
inception_model = Sequential()
inception_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
inception_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
inception_model.add(Dropout(0.4))
inception_model.add(Dense(dog_breeds, activation='softmax'))

#Compiling the Model
inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


#Now train the Model
# checkpointer = ModelCheckpoint(filepath='weights.best.InceptionV3.hdf5',
#                                verbose=1, save_best_only=True)
#
# inception_model.fit(train_InceptionV3, train_targets,
#           validation_data=(valid_InceptionV3, valid_targets),
#           epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)



#Load the Model with the
inception_model.load_weights('weights.best.InceptionV3.hdf5')

#Prints out the summary of the Model
# inception_model.summary()
#
#
#Model Architecture
#
#



#
#
#   Testing the Model on the test Dataset of dog images
#
#
# get index of predicted dog breed for each image in test set
InceptionV3_predictions = [np.argmax(inception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

# report test accuracy
test_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

#
#
#   Predict Dog Breed using the model
#
#
def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))



# top_N defines how many predictions to return
top_N = 4



def predict_breed(path):
    # load image using path_to_tensor
    print('Loading image...')
    image_tensor = path_to_tensor(path)

    # obtain bottleneck features using extract_InceptionV3
    print('Extracting bottleneck features...')
    bottleneck_features = extract_InceptionV3(image_tensor)

    # feed into top_model for breed prediction
    print('Feeding bottlenneck features into top model...')
    prediction = inception_model.predict(bottleneck_features)[0]

    # sort predicted breeds by highest probability, extract the top N predictions
    breeds_predicted = [dog_names[idx] for idx in np.argsort(prediction)[::-1][:top_N]]
    confidence_predicted = np.sort(prediction)[::-1][:top_N]

    print('Predicting breed...')
    # take prediction, lookup in dog_names, return value
    return breeds_predicted, confidence_predicted
#
#
#   Predict Dog Breed using the model
#
#




#
#
#   Our Algorithm for making prediction
#
#
def make_prediction(path, multiple_breeds=False):
    breeds, confidence = predict_breed(path)
    img = mpimg.imread(path)
    plt.axis('off')

    # since the dog detector worked better, and we don't have
    # access to softmax probabilities from dog and face detectors
    # we'll first check for dog detection, and only if there are no dogs
    # detected we'll check for humans
    
    if dog_detector(path):
        if multiple_breeds:
            return breeds, confidence, "dog"        # returns a tuple of ([breeds],[confidence], type of species)

        elif face_detector(path):
            return breeds, confidence, "human"
"""
    if dog_detector(path):
        print('Woof woof!')
        imgplot = plt.imshow(img)
        print('You look like a {}.'.format(breeds[0].replace("_", " ")))
        plt.show()

        if multiple_breeds:
            print('\n\nTop 4 predictions (for mixed breeds)')
            for i, j in zip(breeds, confidence):
                print('Predicted breed: {} with a confidence of {:.4f}'.format(i.replace("_", " "), j))

    elif face_detector(path):
        print('Hello human!')
        imgplot = plt.imshow(img)
        print('If you were a dog, you\'d be a {}.'.format(breeds[0].replace("_", " ")))
    else:
        raise ValueError('Could not detect dogs or humans in image.')
"""
#
#
#   Our Algorithm for making prediction
#
#
