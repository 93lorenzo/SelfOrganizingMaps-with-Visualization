from SOMImplementation import SOMLorenzo
import numpy as np
import sys
import os
from PIL import Image

def RGB_calculus(image):
    image=Image.open(image)
    w = image.size[0]
    h = image.size[1]

    tuple_list = []
    R_M = G_M = B_M =  0
    current_pix = image.load()

    # sum all values
    for i in range(w):
        for j in range(h):
            R_M += current_pix[i, j][0]
            G_M += current_pix[i, j][1]
            B_M += current_pix[i, j][2]
    # divide for all the number of values to have the mean
    R_M = float(R_M) / float(w * h)
    G_M = float(G_M) / float(w * h)
    B_M = float(B_M) / float(w * h)

    tup = {"r_m": R_M, "g_m": G_M, "b_m": B_M}
    tuple_list.append(tup)

    return tuple_list


# per each input_vectors_images_path passed we create and input_vectors list
def making_input_list(input_vectors_images_path,desired_feature,input_vectors):
    for i in range(len(input_vectors_images_path)):

        sys.stdout.write("\r{0}/{1}".format( i, len(input_vectors_images_path) ) )
        sys.stdout.flush()

        vectors = RGB_calculus(input_vectors_images_path[i])
        feature_list = []

        if desired_feature == 3:
            feature_list.append(vectors[0]["r_m"])
            feature_list.append(vectors[0]["g_m"])
            feature_list.append(vectors[0]["b_m"])

            input_vectors[i] = feature_list

    return input_vectors

def images_list_maker(path_folder):
    images_list = []
    slash= "/"
    # to parse correctly the string
    if path_folder[0] != slash:
        path_folder = slash + path_folder

    # for each folder I have all the list of the images
    for folder in os.listdir(os.getcwd() + path_folder):
        p = os.getcwd() + path_folder + slash + folder + slash
        for s in os.listdir(os.getcwd() + path_folder + slash + folder):
            images_list.append(p + s)

    return images_list



# THE DIRECTORY WHERE THE IMAGES ARE
path_dir = "small_path"
# THE LEARNING RATE OF THE SELF ORGANIZING MAPS
LR = 0.6
# MAXIMUM NUMBER OF ITERATION OF THE SELF ORGANIZING MAPS
MAX_ITERATION = 100
# DIMENSION OF THE LATTICE/MAP OF THE SELF ORGANIZING MAPS
# I WOULD RECCOMEND SAME DIMENSION FOR BOTH
w=h=32
# HOW MANY FEATURES? -> RGB MEAN HAS ONLY THREE
# REMEMBER IF YOU CHANGE THAT TO CHANGE ALSO THE LENGTH OF THE MAP WEIGHT
desired_feature = 3 #3  RGB


print "---CREATING INPUT VECTOR---"

input_vectors_images_path = images_list_maker(path_dir)
input_vectors = np.array( [[0 for x in range( desired_feature )] for y in range(len(input_vectors_images_path))])
input_vectors = making_input_list(input_vectors_images_path=input_vectors_images_path,desired_feature=desired_feature,input_vectors=input_vectors)

print "---INPUT VECTOR DONE---"

MyMOP = SOMLorenzo(input_vectors=input_vectors,nodes_weights_w=w,nodes_weights_h=h ,LEARNING_RATE_ZERO=LR, MAX_ITERATION=MAX_ITERATION,vector_len=desired_feature,input_vectors_images_path=input_vectors_images_path)
# THIS IS THE VIDEO NAME TO BE ABLE TO IDENTIFY IT BETTER I CALLED IT USING THE PARAMETERS
video_name =  "len=" + str(len(input_vectors)) + "_iterations=" + str(MAX_ITERATION) + "_w=h=" + str(w) + "_LR" + str(LR) + "_features=" + str(desired_feature) + ".avi"
MyMOP.training(video_name)
