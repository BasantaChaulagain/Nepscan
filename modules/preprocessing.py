import os
import numpy as np

from PIL import Image
from sklearn import preprocessing

def convert_image(image):
    """Reads the image from the given location and then converts it into
    grayscale and also changes its shape to 28 X 28.
    Keyword Arguments:
    image -- relative location of the image
    """
    new_image = Image.open(image)
    new_image = new_image.convert('L')
    new_image = new_image.resize((28, 28), resample=0)
    return new_image


def get_label_encoder(all_labels):
    """Reads the unique categorica labels present in the directory and then
    fits and returns a label encoder that can then encode the categorical
    labels into numerical.
    Keyword Arguments:
    all_labels -- list of categorical labels present
    """
    LABEL_ENCODER = preprocessing.LabelEncoder()
    LABEL_ENCODER = LABEL_ENCODER.fit(np.unique(all_labels))
    return LABEL_ENCODER
    # labels = LABEL_ENCODER.transform(all_labels)
    # return labels


def load_data(data_directory):
    """Loads data from the specified directory
    Keyword Arguments:
    data_directory -- the directory containing train and test directory
    """
    directories = [directory for directory in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory,
                                                 directory))]
    labels = []
    image_location = []
    for directory in directories:
        label_directory = os.path.join(data_directory, directory)
        file_names = [os.path.join(label_directory, a_file)
                      for a_file in os.listdir(label_directory)
                      if a_file.endswith(".jpg")]

        for a_file in file_names:
            image_location.append(a_file)
            labels.append(directory)

    LABEL_ENCODER = get_label_encoder(labels)
    labels = LABEL_ENCODER.transform(labels)

    IMAGES28_INITIAL = []
    for image in image_location:
        new_image = convert_image(image)
        IMAGES28_INITIAL.append(new_image)

    IMAGES28 = []
    for image in IMAGES28_INITIAL:
        temp_image = np.array(image)
        IMAGES28.append(temp_image)

    images = IMAGES28
    images = np.asarray(images)
    labels = np.asarray(labels)
    IMAGES = []
    for image in images:
        image = np.reshape(image, (28, 28, 1))
        IMAGES.append(image)
    images = IMAGES
    images = np.asarray(images)
    images = images.astype(np.float32)
    labels = labels.astype(np.int32)
    return (images, labels)
