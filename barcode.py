# -*- coding: utf8 -*-
# Search barcode on image

import os  # import pac. and lib.
import sys
import cv2
import cProfile
import re

import numpy as np


def read_photo(path_to_image):  # read photo
    '''
    :param path_to_image:This function open the foto and use parameter path_to_image.
    If path or format of the photo incorrect function return warning.
    :return: Function return the read photo.
    '''
    if os.path.isfile(path_to_image):
        try:
            image_test = cv2.imread(path_to_image, 0)
        except Exception as error:
            print('Format is not wrong')
        else:
            return image_test
    else:
        print('Path is not exist')


def filter_image(image_test):
    '''
    :param image_test: This function use parameter image_test.
    Use operation the blur-filter and the binarisation.
    :return: Function return image threshold in range 150:255
    '''
    blur = cv2.blur(image_test.copy(), (3, 3))
    ret, image_thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
    return image_thresh


def contours_create(image_thresh, image_test):
    '''
    :param image_test: image_thresh, image_test
    This function create contours of barcode.
    :return: Function return variable barzones which draw rectangle on top of the barcode.

    '''

    contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = np.array([])  # create array of perimeters
    for item in contours:
        perimeter = np.append(perimeter, cv2.arcLength(item, True))  # add elements of contours array
    image_unique = np.unique(perimeter, return_counts=True)  # unique elements
    image_unique_mean = image_unique[0].mean()  # count of mean
    image_unique_diap = np.logical_and(image_unique[0] < image_unique_mean * 1.7,
                                       image_unique[0] > image_unique_mean * 0.3)  # create dispersion +- 70%(mean)
    index = np.array([])
    for i in image_unique[0][image_unique_diap]:
        perimeter_index = np.take(contours, np.where(perimeter == i))
        index = np.append(index, perimeter_index)
    main_contour = np.array([])  # create contours array for calculate main contours

    for contours_index in index:
        main_contour = np.append(main_contour, contours_index)

    main_shape = main_contour.shape[0] / 2
    main_shape = int(main_shape)  # converted int
    main_contour = main_contour.reshape(main_shape, 2)  # create 2-D coordinate array
    main_contour = main_contour.astype(int)  # convert int
    x1 = main_contour.min(axis=0)[0]  # calculate rectangle coordinate
    y1 = main_contour.max(axis=0)[1]
    x2 = main_contour.max(axis=0)[0]
    y2 = main_contour.min(axis=0)[1]
    barzones = cv2.rectangle(image_test, (x1.astype(int), y1.astype(int)), (x2.astype(int), y2.astype(int)),
                             (0, 0, 255), 2)

    return barzones


def main_algorithm(**kwargs):
    '''Main algoritm'''
    path_to_image = kwargs['path']
    image_original = read_photo(path_to_image)
    fil = filter_image(image_original)
    image_with_contours = contours_create(fil, image_original)
    key = kwargs['key']
    path_to_save = 'photo/save/out.png'

    if key == 'o':
        cv2.imwrite(path_to_save, image_with_contours)  # save image
    elif key == 'i':
        cv2.imshow("Barcode image", image_with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif key == 't':
        cProfile.run('re.compile("main_algorithm")')


if __name__ == '__main__':

    print(sys.argv)
    try:
        path_to_image = sys.argv[1]
        key = None
        path_to_save = None
        if len(sys.argv) >= 3:

            key = sys.argv[2]
            if key not in ['i', 'o', 't']:
                assert False, 'Error'
            if key == 'o':
                try:
                    path_to_save = 'photo/save/out.png'
                except:
                    print('use default')



    except Exception as error:
        print(error)
        raise AttributeError

    main_algorithm(path=path_to_image, key=key, path_to_save=path_to_save)
