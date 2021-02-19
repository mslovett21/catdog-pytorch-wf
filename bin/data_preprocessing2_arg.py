#!/usr/bin/env python3

"""
MACHINE LEARNING WORKFLOWS - STEP 2 - DATA PREPROCESSING (might be a multistep process of itself)

Data  Augmentation. 

Here we augment the images.


"""
import numpy as np
import glob, os
import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='Data Preprocessing step 2')
parser.add_argument('dataset_name',  metavar='dataset', type=str, nargs=1, help = "training set prefix")

IMG_SIZE = 150
DATASET_DIR = ""

class DataAugmentation:
    gaussian = np.random.normal(0, 0.3, (IMG_SIZE ,IMG_SIZE,3 ) )
    

    def load_img(self,img_path):
        return cv2.imread(img_path)


    def rotate(self, image, angle=20, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image


    def add_gaussian_noise(self, img):
        return img + self.gaussian

    
    
    def image_augment(self, path): 
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        ''' 
        img = self.load_img(path)
        img = img/255
        img_scale = self.rotate(img, 10, 1.7)
        img_rot = self.rotate(img,20, 1.2)
        img_gaussian = self.add_gaussian_noise(img)
        return [img, img_scale, img_rot, img_gaussian]

def main():
    augmentation = DataAugmentation()
    args  = parser.parse_args()
    dataset = args.dataset_name[0]

    for file in glob.glob( DATASET_DIR + "*proc1.jpg"):
        augmented_imgs = augmentation.image_augment(file)
        label = file.split("/")[-1].split("_")[0:2]
        name = dataset + "_" + "_".join(label) + "_proc2_"
        i = 0
        for img in augmented_imgs:
            fname = DATASET_DIR + name + str(i) + '.jpg'
            cv2.imwrite(fname,img*255)# if you want to see the images do img*255
            i +=1



if __name__ == "__main__":
    main()
