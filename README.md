# code for stroke subtyping

* main function: main.py
In the main function, data will be loaded from the scratch file system.
Two residual nets were designed. one is to learn the global features, another is to learn the details from images.
(optional) second order gradient decent function has been designed for optimization. 

* cnn3d.py a simple 3D CNN architecture for comparison use.

* weighted_crossentropy.py  weighted cross entropy functions to deal with class inbalance.

lib folder contains functions to process the data, generate batches, visualize gradient maps, restore the trained model, and other functions.