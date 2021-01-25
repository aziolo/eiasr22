import cv2 as cv
import numpy
import os
import re

import face_extractor


class Codebook:
    """Codebook conatins larened feature vectors saved in txt files,
        each feature vector consist of calculated HOG for image,
        txt filenames in codebook corresponds to image file from learning set """

    def __init__(self):
        '''
        constructor function for the class
        '''
        self.templates_list = []
        self.ext_list = ['.jpg', '.bmp', '.png']

    def create_codebook(self, in_path='cropped_images', out_path='Codebook'):
        '''
        Main function of the class to prepare codebook
        Function loads in sequence all preprocesses image files from specified in_path,
        and create feature vector for each image

        :param in_path root directory for all the cropped images prepared for learning
        :param out_path root directory to store codebook
        '''

        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        img_paths = self._get_img_paths(in_path)
        iter = 0  # to test purpose
        print("Creating codebook...")

        # Initial call to print 0% progress
        l = len(img_paths)
        i = 0
        printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

        for img in img_paths:
            hog_features_vect = self.Calc_descriptors(img)
            dir_list = img.split(os.sep)
            file_name = dir_list[-1]  # get filename of jpeg image file
            file_name = os.path.splitext(file_name)[0] + ".npy"  # change extension
            outfile = os.path.join(out_path, file_name)
            numpy.save(outfile, hog_features_vect)
            # time.sleep(0.1)
            # Update Progress Bar
            i = i + 1
            printProgressBar(i, l, prefix='Progress:', suffix='Complete', length=50)
        print("Codebook created")

    def _get_img_paths(self, in_path):
        '''
        get paths of all images in root directory 
        (specific to the current dir structure!)
    
        :return img_paths list of paths of pictures to learn
        '''

        dir_list = os.listdir(in_path)
        img_paths = []

        for directory in dir_list:
            directory = os.path.join(in_path, directory)
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

    def _get_template_paths(self, codebook_path):
        '''
        get paths of all templates in codebook 
        (specific to the current dir structure!)
    
        :return template_paths list of paths of templates with calculated HOGs
        '''

        dir_list = os.listdir(codebook_path)
        template_paths = []

        for directory in dir_list:
            directory = os.path.join(codebook_path, directory)
            template_paths.append(directory)

        return template_paths

    def get_img_data(self, img_path):
        '''
        get information from dtext filename,
        returns dictionary with image vesrsion, person number, series number, file number 
        in directory, vertical and horizontal angle in degrees

        :return img_data dictionary with img data 
        '''

        img_data = {
            'image_version': 0,
            'person_number': 0,
            'series_number': 0,
            'file_number': 0,
            'vertical': 0,
            'horizontal': 0
        }
        filename = img_path.split(os.sep)[-1]
        data = re.findall("\d*\d", filename)
        signs = re.findall('[+ -]', filename)
        img_data['image_version'] = int(data[0])
        img_data['person_number'] = int(data[1][0:1])
        img_data['series_number'] = int(data[1][2])
        img_data['file_number'] = int(data[1][3:4])
        img_data['vertical'] = \
            int(data[2]) if signs[0] == '+' else -int(data[2])
        img_data['horizontal'] = \
            int(data[3]) if signs[1] == '+' else -int(data[3])

        return img_data

    def get_template_data(self, template_path):
        '''
        Get angle values from txt template filename

        :param template_path - path to txt file in codebook dir
        :return v_angle & h_angle obtained from template filename
        '''

        v_angle = 0;
        h_angle = 0
        filename = template_path.split(os.sep)[-1]
        data = re.findall("\d*\d", filename)
        signs = re.findall('[+ -]', filename)
        v_angle = int(data[2]) if signs[0] == '+' else -int(data[2])
        h_angle = int(data[3]) if signs[1] == '+' else -int(data[3])

        return v_angle, h_angle

    def Calc_descriptors(self, img_path):
        '''
        Calculate descriptors (HOG) for an image and return feature vector

        :param img_path directory to image for wich calculating descriptors

        :return hog_features as numpy array - our feature vector for this image
        '''

        img = cv.imread(img_path)
        cell_size = (32, 32)  # w x h in pixels
        block_size = (4, 4)  # w x h in cells
        nbins = 9  # number of orientation bins
        hog_desc = cv.HOGDescriptor(_winSize=(img.shape[0] // cell_size[0] * cell_size[0],
                                              img.shape[1] // cell_size[1] * cell_size[1]),
                                    _blockSize=(block_size[0] * cell_size[0],
                                                block_size[1] * cell_size[1]),
                                    _blockStride=(cell_size[0], cell_size[1]),
                                    _cellSize=(cell_size[0], cell_size[1]),
                                    _nbins=nbins)
        hog_features = hog_desc.compute(img)
        print(len(hog_features))
        return hog_features

    def Estimate_angles_for_img(self, test_img_path):
        '''
        Calculate descriptors (HOG) for test image, finding best matching
        feature vector in Codebook and based on that information estimating angles

        :return verical_angle & horizontal_angle estimated for input image
        '''

        distance = 100000
        vertical_angle = 0
        horizontal_angle = 0

        test_img_hog = self.Calc_descriptors(test_img_path)
        print("Estimating orientation for: " + test_img_path)

        # Initial call to print 0% progress
        l = len(self.templates_list)
        i = 0
        printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

        for template in self.templates_list:
            if not (test_img_hog.shape[0] == template['hog'].shape[0]):
                continue
            temp = numpy.linalg.norm((template['hog'] - test_img_hog))
            if (temp < distance):
                distance = temp
                vertical_angle, horizontal_angle = template['vertical'], template['horizontal']

            # Update Progress Bar
            i = i + 1
            printProgressBar(i, l, prefix='Progress:', suffix='Complete', length=50)

        return vertical_angle, horizontal_angle

    def Load_codebook_to_mem(self, codebook_path='Codebook'):
        '''
        Load codebook templates to memory for later estimating head orientation

        :param codebook_path directory to image for estimate face orientation 
        '''
        template_paths = self._get_template_paths(codebook_path)

        print("Loading Codebook from files...")
        # Initial call to print 0% progress
        l = len(template_paths)
        i = 0
        printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

        for template_path in template_paths:
            template = {
                'hog': 0,
                'vertical': 0,
                'horizontal': 0
            }
            template['hog'] = numpy.load(template_path)
            template['vertical'], template['horizontal'] = \
                self.get_template_data(template_path)
            self.templates_list.append(template)
            # Update Progress Bar
            i = i + 1
            printProgressBar(i, l, prefix='Progress:', suffix='Complete', length=50)

    # Print iterations progress
    # Code from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console


def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()