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
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

bottleneck_features = np.load('DogInceptionV3Data.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']
train_InceptionV3.shape[1:]

inception_model = Sequential()
inception_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
inception_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
inception_model.add(Dropout(0.4))
inception_model.add(Dense(dog_breeds, activation='softmax'))

#Compiling the Model
inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


#Load the Model with the
inception_model.load_weights('weights.best.InceptionV3.hdf5')

#
# get index of predicted dog breed for each image in test set
InceptionV3_predictions = [np.argmax(inception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

# report test accuracy
test_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

#Takes most significants breeds (that is, we show breeds until we've
#reach a certain threshold (2%))
#Then we just fill out the rest of the graph with the 'other' label
#That way we have a complete pie graph (100% percent) 
def topBreeds(breeds, conf):
    dogs = []
    per = []
    for i in range(0, len(breeds)):
        if conf[i] < 0.02:
            break
        per.append(conf[i])
    #find out the value of 'other'
    total = np.sum(per)
    other = 1 - total
    
    for i in range(0, len(per)):
        answer = breeds[i].replace('_', ' ')
        dogs.append(answer)
        
    per.append(other)
    dogs.append('Other')
    return dogs, per

# top_N defines how many predictions to return
top_N = 10 

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

	#Get top breeds, upto 10 significant breeds
    breeds, percentage = topBreeds(breeds_predicted, confidence_predicted)
    print('Predicting breed...')
    # take prediction, lookup in dog_names, return value
    return breeds, percentage
#
#
#   Predict Dog Breed using the model
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
