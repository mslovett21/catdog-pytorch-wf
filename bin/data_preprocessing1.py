#!/usr/bin/env python3

"""
MACHINE LEARNING WORKFLOWS - STEP 2 - DATA PREPROCESSING (might be a multistep process of itself)

Data  preprocessing is essential for building a successful ML model. 

Here we resize the images.


"""
import glob, os
import cv2
DATASET_DIR = ""


class ImagePreprocessor:   
    IMG_SIZE = 150
    
    def load_img(self,img_path):
        return cv2.imread(img_path)
    
    def resize_img(self,img):
        return cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
    
    def basic_preprocessing(self,img_path):
        img = self.load_img(img_path)
        img = self.resize_img(img)
        return img
    

def main():
    image_preprocessor = ImagePreprocessor()
    invalid_files = []

    for file in glob.glob(DATASET_DIR + "*.jpg"):
        try:
            img = image_preprocessor.basic_preprocessing(file)
            curr_label = file.split('.')[0]
            cv2.imwrite(curr_label + '_proc1.jpg',img)
        except:
            invalid_files.append(file)
            print(file)
'''
# TO CATCH ALL the corrupted files that would be an issue for Pegasus
    f =  open("corrupted_files.txt", "w")
    for file in invalid_files:
        f.write(file)

'''
if __name__ == "__main__":
    main()
