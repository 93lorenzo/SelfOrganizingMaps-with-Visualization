import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from matplotlib.lines import Line2D
import sys
from time import gmtime, strftime


class SOMExample(object):

    ## Self Organizing Map Class
    ## self.input_vectors -> all the "training data" given in input to move the nodes
    ## self.nodes_weights_h_w -> height, weight of the bidimensional map
    ## self.nodes_weights -> the map containing all the weights the default should be h * w *vector_len(=3)
    ## self.LEARNING_RATE_ZERO -> starter learning rate
    ## self.MAX_ITERATION -> how many times it has to iterate
    ## self.T -> it indicated the current iteration of the map training
    ## self.vector_len -> it indicated the size of the vector default is 3 -> R G B
    def __init__(self, input_vectors, nodes_weights_h,nodes_weights_w, LEARNING_RATE_ZERO=0.6, MAX_ITERATION=5, T=0, vector_len=3, input_vectors_images_path=None) :
        self.input_vectors = input_vectors
        print "input vectors shape"
        print input_vectors.shape
        self.input_vectors_images_path = input_vectors_images_path
        self.nodes_weights_h = nodes_weights_h
        self.nodes_weights_w = nodes_weights_w
        self.vector_len = vector_len

        self.nodes_weights = np.random.rand(self.nodes_weights_h,self.nodes_weights_w,self.vector_len)
        self.LEARNING_RATE_ZERO = LEARNING_RATE_ZERO
        self.LATTICE_WIDTH_ZERO = max(self.nodes_weights_h,self.nodes_weights_w) /2
        self.MAX_ITERATION = MAX_ITERATION
        # radius and learning rate depend on the current iteration step
        self.CURRENT_LATTICE_WIDTH = max(self.nodes_weights_h,self.nodes_weights_w) /2  #self.LATTICE_WIDTH_ZERO
        self.CURRENT_LEARNING_RATE = self.LEARNING_RATE_ZERO
        # always start from 0
        self.T = T

    # function used to calculate the euclidean distance
    @staticmethod
    def EuclideanDistance( A,B):
        sum = 0
        if not len(A) == len(B):
            print "--- ERROR VECTORS HAVE DIFFERENT LENGTH ---"
            return None
        else :
            for i in range(len(A)) :
                sum += ( np.array(A[i]) - np.array(B[i]) ) **2
            return math.sqrt(sum)


    #to obtain the index of the closer input vector to the x y coordinates of the weight
    def most_like_weight(self, x,y):
        x = np.int(x)
        y = np.int(y)
        current_min = 10000000
        current_ind = -1

        for i  in  range (len(self.input_vectors)) :
            if self.EuclideanDistance(self.input_vectors[i],self.nodes_weights[x,y]) < current_min :
                current_min = self.EuclideanDistance(self.input_vectors[i],self.nodes_weights[x,y])
                current_ind = i
        return current_ind

    ## Every node is examinated to calculate which one's weights are
    ## most like the input vector. (MIN DISTANCE)
    ## The Best Matching Unit is the winning vector
    def most_like_vector(self, input_vector):
        dists = (self.nodes_weights - input_vector)**2
        # No need to do sqrt here since we are looking for a minimum not an exact euclidean distnace
        # dists = np.sqrt(dists.sum(1))
        dists = dists.sum(2)
        min_dist_node = np.where(dists == dists.min())
        return min_dist_node[0][0], min_dist_node[1][0]

    ## Radius(=CURRENT_LATTICE_WIDTH) of Neighbourhood is calculated (within radius = Best Matching Unit neighbourhood)
    ## It starts large and then decrease each step with EXPONENTIAL DECAY
    def radius_and_learning_rate_update(self):
        # LEARNING_RATE with exponential decay
        time_constant = float(self.MAX_ITERATION)/float(math.log(self.CURRENT_LATTICE_WIDTH))
        exp = math.e ** (- float(self.T) / float(time_constant))
        self.CURRENT_LEARNING_RATE = self.LEARNING_RATE_ZERO * exp
        self.CURRENT_LATTICE_WIDTH = self.LATTICE_WIDTH_ZERO * exp


    ## Each neighbouring node's weights are adjusted to make them more like the input vector.
    ## The closer a node to the BMU is, and the more is moved closer to the BMU
    def weights_adjusting(self,current_i, current_j,input_vector ):
        h, w, c = self.nodes_weights.shape
        y, x = np.ogrid[-current_i:h - current_i, -current_j:w - current_j]
        # take the values within the radius (CURRENT_LATTICE_WIDTH)
        mask = x * x + y * y <= self.CURRENT_LATTICE_WIDTH**2
        # w -> weights we have to change
        w = self.nodes_weights[mask]
        best = self.nodes_weights[current_i][current_j]
        DISTANCE_INFLUENCE = math.e ** ( (-(best-w)) / ((2*self.CURRENT_LATTICE_WIDTH**2)))
        # vector update
        self.nodes_weights[mask] = w + self.CURRENT_LEARNING_RATE * DISTANCE_INFLUENCE *(input_vector - w)

    ## Function used to get the images we will use to make the video
    def get_map_image(self, resolution):
        ns_img = None
        ns_img = cv2.normalize(self.nodes_weights[:,:,:3], ns_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        ns_img = cv2.resize(ns_img, resolution)
        return ns_img


    def final_plot(self):
        w = self.nodes_weights_w
        h = self.nodes_weights_h

        # Generate data x, y for scatter and an array of images.
        x = np.arange(w)
        y = np.arange(h)
        # an image array as big as the som lattice
        # fixed values for the shape of the image and the arrow so it will not be too small
        fixed_h = fixed_w = 120

        images_array = np.empty((len(self.input_vectors), fixed_w, fixed_h, 3))
        # associate for each index of the images_array an image that should be a input vector data
        for i in range(len(self.input_vectors)):
            # save the images in the images_array also resizing as necessary (images_array dimensions)
            images_array[i] = np.array(cv2.resize(cv2.imread(self.input_vectors_images_path[i]), (fixed_h, fixed_w)))

        y = np.zeros((w, h))
        x = np.zeros((w, h))
        # VERY IMPORTANT
        # WE WANT A SQUARE SHAPE nxn
        # VERY IMPORTANT
        for i in range(len(x)):
            for j in range(len(x[0])):
                x[i][j] = j
                y[i][j] = i
        # to obtain all the elements of the lattice ordered in one dimensional vector
        x = np.array(x.reshape(len(x) * len(x[0]), 1))
        y = np.array(y.reshape(len(y) * len(y[0]), 1))

        fig = plt.figure()
        # just imposing how the plot should be 1x1
        ax = fig.add_subplot(111)
        line = Line2D(x, y, ls="", marker="")
        ax.add_line(line)

        # upper is important, [:,:,:3] the colors are done by only the first three components
        # ******************************************************
        # NORMALIZED WEIGHTS
        a = self.nodes_weights[:,:,:3]
        normalized_weights =  (255*(a - np.max(a))/-np.ptp(a)).astype(np.uint8)
        ax.imshow( np.invert( normalized_weights ) , origin='upper')

        # lines = plt.scatter( x, y,  marker="+")
        # create the annotations box
        im = OffsetImage(images_array[0, :, :, :], zoom=1)
        #xybox = (50., 50.) # this is also for the arrow
        xybox = (fixed_h, fixed_w)  # this is also for the arrow
        ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data', boxcoords="offset points", pad=0.3,
                            arrowprops=dict(arrowstyle="->"))
        # add it to the axes and make it invisible
        ax.add_artist(ab)
        ab.set_visible(False)

        def hover(event):

            if line.contains(event)[0]:
                # find out the index within the array from the event
                ind = line.contains(event)[1]["ind"]
                # get the figure size
                ww, hh = fig.get_size_inches() * fig.dpi
                ws = (event.x > ww / 2.) * -1 + (event.x <= ww / 2.)
                hs = (event.y > hh / 2.) * -1 + (event.y <= hh / 2.)
                # if event occurs in the top or right quadrant of the figure,
                # change the annotation box position relative to mouse.
                ind = ind[0]
                ab.xybox = (xybox[0] * ws, xybox[1] * hs) # this should be the lenght of the arrow
                #ab.xybox = (xybox[0] * ws, xybox[1] * hs)
                # make annotation box visible
                ab.set_visible(True)
                # place it at the position of the hovered scatter point
                ab.xy = (x[ind], y[ind])
                # set the image corresponding to that point
                # ******
                # TO OBTAIN THE VALUES OF THE LATTICE
                x_index = math.floor(int(ind) / h)
                y_index = math.floor(int(ind) - (h * math.floor(int(ind) / h)))
                # take the correct "label" index
                image_index = self.most_like_weight(x_index, y_index)
                # im.set_data(  images_array[ind ,:,:][0] )
                #print "image_index = " + str ( image_index )
                normalized_weights_im = images_array[image_index, :, :]
                normalized_weights_im = (255 * (normalized_weights_im - np.max(normalized_weights_im)) / -np.ptp(normalized_weights_im)).astype(np.uint8)

                # VERY IMPORTANT
                # it has to be that way with invert and [:,:,[2,1,0] from opencv to matplotlib
                # VERY IMPORTANT
                im.set_data(np.invert( normalized_weights_im )[:,:,[2,1,0]])
                # important to debug
                #print self.input_vectors_images_path[image_index]
            else:
                # if the mouse is not over a scatter point
                ab.set_visible(False)

            fig.canvas.draw_idle()

        # add callback for mouse moves
        fig.canvas.mpl_connect('motion_notify_event', hover)
        # fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def training(self,video_file=None):

        frame_resolution = (512, 512)  # Resolution of the video frames and the som.jpg image with the progress
        video = None

        actual_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        directory = ("images/" + str(actual_time) + "/").replace(":", "-")
        print directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        if video_file is not None:
            # depending on the versions
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc =cv2.cv.CV_FOURCC(*'XVID')

            video = cv2.VideoWriter(directory+video_file, fourcc, 20, frame_resolution)


        # epoch loops I assumed that the batch size is the same of the input data so with one iteration I do one epoch
        while self.T < self.MAX_ITERATION :
            # if the lattice width is one it is moving just one node no influences on the others -> useless to continue
            if self.CURRENT_LATTICE_WIDTH <= 1.0:
                break

            i = 0
            for x in self.input_vectors:
                if i % (len(self.input_vectors)/10) == 0 or i == len(self.input_vectors) - 1 :
                    sys.stdout.write("\r{0}/{1} {2}>".format( (i+1), len(self.input_vectors), "=" * ( i / (len(self.input_vectors)/10)  )))
                    sys.stdout.flush()
                i = i + 1

                # TAKE THE BEST
                curr_i, curr_j = self.most_like_vector(x)
                # ADJUST THE WHEIGHTS AND UPDATE
                self.weights_adjusting(curr_i,curr_j,x)

            print " TRAINING ITERATION " + str(self.T) + " LR = " + str(self.CURRENT_LEARNING_RATE) + " width " + str(self.CURRENT_LATTICE_WIDTH)

            img = self.get_map_image(frame_resolution)
            cv2.imshow('SOM', img[:,:,:3])
            cv2.waitKey(1)

            if video is not None:
                video.write(img)

            # update everything
            self.T += 1
            self.radius_and_learning_rate_update()
            if self.T % 5 == 0 :
                cv2.imwrite(directory+"T_="+str(self.T)+".png", img)

        # Close the video file
        if video is not None:
            video.release()

        #Do the final image with the closest image to each lattice point
        self.final_plot()


