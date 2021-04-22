# SelfOrganizingMaps-with-Visualization
python 2 SelfOrganizingMaps Implementation

- Basic Implementation of a self organizing maps
There are two files : 
****************
SOMTest.py 
****************
It contains the parameters that should be changed due to what fits you best: 

 THE DIRECTORY WHERE THE IMAGES ARE
 
path_dir = "small_path"

THE LEARNING RATE OF THE SELF ORGANIZING MAPS

LR = 0.6

MAXIMUM NUMBER OF ITERATION OF THE SELF ORGANIZING MAPS
MAX_ITERATION = 100

DIMENSION OF THE LATTICE/MAP OF THE SELF ORGANIZING MAPS
I WOULD RECCOMEND SAME DIMENSION FOR BOTH

w=h=32

HOW MANY FEATURES? -> RGB MEAN HAS ONLY THREE
 REMEMBER IF YOU CHANGE THAT TO CHANGE ALSO THE LENGTH OF THE MAP WEIGHT

desired_feature = 3 #3  RGB

After that from the directory specified in path_dir variable it will be able to create the input vectors list
and consequently to create an object SOMLorenzo with all the parameter specified

Then there is the training of the Map, passing as a name the name for the video of all the map evolution during the time.

Images and video will be stored in a new folder created during the training.

Of course you can change the parameters number and how to extract them in this case I only extract the mean value of the 
RGB Colors of each image.


****************
SOMImplementation.py 
****************
This file contains the class SOMLORENZO 
here is where the self organizing maps is implemented.


[Video Example](https://www.youtube.com/watch?v=r-MkZm0s2iM)
