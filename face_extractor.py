import cv2 as cv
import numpy
import os
import re

from numpy.core.multiarray import ndarray


'''
get information from image filename

returns dictionary with person number, series number, file number 
    in directory, vertical and horizontal angle in degrees

:return img_data dictionary with img data 
'''


def get_img_data(img_path):
    img_data = {
        'person_number': 0,
        'series_number': 0,
        'file_number': 0,
        'vertical': 0,
        'horizontal': 0
    }
    filename = img_path.split(os.sep)[-1]
    data = re.findall("\d*\d", filename)
    signs = re.findall('[+ -]', filename)
    img_data['person_number'] = int(data[0][0:1])
    img_data['series_number'] = int(data[0][2])
    img_data['file_number'] = int(data[0][3:4])
    img_data['vertical'] = \
        int(data[1]) if signs[0] == '+' else -int(data[1])
    img_data['horizontal'] = \
        int(data[2]) if signs[1] == '+' else -int(data[2])

    return img_data


'''
get roi coordinates with face of corresponding image

:param img_path path to file to get face coordinates from

:return faces ndarray with x, y, width and height of a face
'''


def get_face_from_metadata(img_path):
    txt_path = img_path[:-3]+'txt'
    faces = ndarray(shape=(1, 4), dtype=int)

    try:
        with open(txt_path, 'r') as in_file:
            file_content = in_file.readlines()
    except FileNotFoundError as e:
        print("Image {} does not have a corresponding .txt file.".format(
            img_path))
        return None

    faces[0] = [int(file_content[-4])-int(file_content[-2])/2,
                int(file_content[-3])-int(file_content[-1])/2,
                int(file_content[-2]),
                int(file_content[-1])]

    return faces


'''
This class uses OpenCV library to find faces in images and preprocess them
'''


class FaceExtractor:

    '''
    constructor function for the class

    :param in_path root directory for all the images to work on in this class
    :param out_path root directory to store preprocessed images in
    '''
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path
        self.cascade_path = {'front': 'haarcascade_frontalface_default.xml',
                             'side': 'haarcascade_sideface.xml'}
        self.face_cascade = cv.CascadeClassifier(self.cascade_path['front'])
        self.ext_list = ['.jpg', '.bmp', '.png']



    '''
    Main function of the class to prepare images before executing 
    the core algorithm in the project
    
    Function loads in sequence all filesfrom specified in_path,
    preprocesses them and saves images to directory specified in out_path.
    '''
    def preprocess_images(self):
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        img_paths = self._get_img_paths()

        for img in img_paths:
            preprocessed_faces = self._preprocess_img(img, use_metadata=True)
            dir_list = img.split(os.sep)
            nested_dir = dir_list[-2]
            file_name = dir_list[-1]
            current_out_dir = os.path.join(self.out_path, nested_dir)
            if not os.path.isdir(current_out_dir):
                os.mkdir(current_out_dir)

            for count, face in enumerate(preprocessed_faces):
                cv.imwrite(os.path.join(current_out_dir,
                                        str(count)+file_name), face)


    '''
    get paths of all images in root directory 
    (specific to the current dir structure!)
    
    :return img_paths list of paths to pictures to preprocess
    '''
    def _get_img_paths(self):
        dir_list = os.listdir(self.in_path)
        img_paths = []

        for directory in dir_list:
            directory = os.path.join(self.in_path, directory)
            if not os.path.isdir(directory):
                continue

            img_list = os.listdir(directory)
            # get rid of all files that aren't images
            img_list = [img for img in img_list if
                        any(ext in img for ext in self.ext_list)]

            # add directory
            img_list = [os.path.join(directory, img) for img in img_list]
            img_paths = img_paths + img_list

        return img_paths


    '''
    finds face, crops the image to match the face, changes color space, scale
    
    :param img_path path to the image to preprocess
    :param use_metadata set if corresponding .txt file exists 
        with face coordinates
    
    :return prep_images list of preprocessed images objects of found faces
    '''
    def _preprocess_img(self, img_path, use_metadata=False):
        img = cv.imread(img_path)
        img_data = get_img_data(img_path)
        desired_size = (128, 128)
        threshold = 30
        prep_images = []

        # flip image, because cascade can recognize only right-facing heads
        if img_data['horizontal'] > threshold:
            img = cv.flip(img, 1)

        # decide what cascade to use
        if abs(img_data['horizontal']) > threshold:
            self.face_cascade = cv.CascadeClassifier(self.cascade_path['side'])
        else:
            self.face_cascade = cv.CascadeClassifier(self.cascade_path['front'])

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if use_metadata:
            faces = get_face_from_metadata(img_path)
        else:
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 3)

        # Draw rectangle around the faces
        if faces is not None:
            for (x, y, w, h) in faces:
                # flip back if head is supposed to be facing left
                if img_data['horizontal'] > threshold:
                    gray = cv.flip(gray, 1)

                roi = gray[y:y+h, x:x+w]
                roi = cv.resize(roi, desired_size)
                prep_images.append(roi)

                # for now we want to find only one face in a picture, for learning
                break

        return prep_images



