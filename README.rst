# ThobePose

.. title:: Thobe Dataset

.. figure::  docs/_static/imgs/header.png
   :align:   center
   :width: 960px



Description
+++++++++++
To kick off a series of future projects aiming to provide optimum estimation of human pose with difficult dress variations, and to compensate for the lacking of training and testing data, we used data synthesis approach discussed in `[1] <http://www.cse.chalmers.se/edu/year/2010/course/TDA361/Advanced%20Computer%20Graphics/BodyPartRecognition.pdf/>`_ to synthesize new dataset for estimating pose of human wearing Thobe. Instead of saving large number of images, and since our approach is based on synthesizing data, we provide in this page the raw material and a recipe needed for compiling and synthesizing the data. By following the recipe below, you will be able to generate the full base dataset as well as generating manipulated data by modifying the material. For simple usage, you can download the sample datasets in `Get Started <#get-started>`_ section below.
 
The datasets that can be generated from the raw material are based on a single man character with a physique that represents a taut body person with a moderate tall. There are maximum of 1193 frames for each dataset described below.
 
Datasets
++++++++
The raw material (as it is without modification) allows you to synthesize the following datasets:
 
Plain-character Dataset
#######################
The plain dataset emulates the dataset in `[2] <http:
//www.citrinitas.com/history of viscom/rockandcaves.html>`_ and the dataset discussed but not released in `[1] <http://www.cse.chalmers.se/edu/year/2010/course/TDA361/Advanced%20Computer%20Graphics/BodyPartRecognition.pdf>`_.
 
.. figure::  docs/_static/imgs/nothobe.png
   :align:   center
   :width: 480px

Thobe-character Dataset
#######################
Initially, we followed the body segmentation provided in `[1] <http://www.cse.chalmers.se/edu/year/2010/course/TDA361/Advanced%20Computer%20Graphics/BodyPartRecognition.pdf>`_ which divides the body into 31 segments. Later, however, we simplified the segmentation by merging the lower body parts into two and four segments.
 
2colors
&&&&&&&
In this dataset, we segment the body into 26 parts. The upper body part is segmented according to definitions in `[1] <http://www.cse.chalmers.se/edu/year/2010/course/TDA361/Advanced%20Computer%20Graphics/BodyPartRecognition.pdf>`_. The lower body part (excluding the feet) however, is divided into two parts only: Left Lower Part, and Right Lower Part. Each of these two segments encompasses the thigh, the knee, the leg, and the ankle.
 
.. figure::  docs/_static/imgs/2colors.png
   :align:   center
   :width: 480px
 
4colors
&&&&&&&
In this dataset, we segment the body into 28 parts. Similar to the previous dataset, we divide the lower body parts in 4 segments: 2 Left and 2 Right. This segmentation is adopted in `[2] <http:
//www.citrinitas.com/history of viscom/rockandcaves.html>`_ as well. This segmentation, in fact, is more appropriate for two reasons. First, the lower body part consists of two big bones on each side: Femur (Thigh Bone), Tibia and Fibula(the leg bones). These bones construct two complete distinctive body parts: thighs and legs and hence can be represented by 4 colors. Second, Thobe hides all the distinctive features of the knee and the ankle, and in many cases (e.g. T-Pose) it is very difficult to estimate their specific locations without considering the geometrical nature and properties of a human being.
 
.. figure::  docs/_static/imgs/4colors.png
   :align:   center
   :width: 480px
 
8colors
&&&&&&&
The 8colors segmentation divides the body into 32 parts. It was first introduced by Shotton in `[1] <http://www.cse.chalmers.se/edu/year/2010/course/TDA361/Advanced%20Computer%20Graphics/BodyPartRecognition.pdf/>`_
 
.. figure::  docs/_static/imgs/8colors.png
   :align:   center
   :width: 480px
 
Pre-requisite
+++++++++++++
The following applications are required in order to generate the datasets:
 
1. `Maya <http://www.autodesk.com/products/maya/overview>`_ from AutoDesk. You can acquire a student license for free
2. `Python 2.7 <https://www.python.org/download/releases/2.7/>`_
 
Get Started
+++++++++++
If you wish to use the basic datasets, please download them from here(`nothobe <https://www.icloud.com/iclouddrive/0OqTDgMj1USBW7f4rTiNnyIcw#nothobe/>`_ & `4colors <https://www.icloud.com/iclouddrive/0uaxHCEOqwouoh129ZoF8AENg#4colors/>`_)
 
To be able to acquire the full dataset, please follow the following steps:
               
1. Clone this repo
   - git clone https://github.com/jalalirs/ThobePose.git
2. Run the renderer
                - cd ThobePose/scripts
                - ./render.sh
4. The renderer will start rendering the scene in batches of 50 frames per run. The rendering will take a while (~3 hours).
5. The rendered depth and ground truth images will be stored in "images/pngs"
6. The script will render the dataset according to the default configuration. Please read below to know what they are and how to change them
 
 
Raw Material
++++++++++++
Below we describe the function of each file provided in the raw material
 
Scripts
#######
* **render.sh**: a bash file to run the renderer in batch mode. The batch size is determined by the capacity of the RAM. With average RAM size (e.g. 16GB), the batch size should not exceed 50. The reason is that as rendering proceed the RAM will be over flowed.
* **render_thobe.py**: a Maya python script to render thobe-character datasets. This script depends on python but must be run with "mayapy". **render.sh** calls this script for rendering.
* **render_plain.py**: a Maya python script to render render-character datasets. This script depends on python but must be run with "mayapy". **render.sh** calls this script for rendering.
* **iff_reader.py**: the previous two scripts use this script to extract png depth and ground images from the iff files generated by Maya Hardware renderer.
* **make_transparent.py**: remove background from ground truth images
* **combine_images.py**: combine multiple ground truth images in one image
 
Scene Files
###########
* **Hamed.mb**: Maya model file.
* **Cache/**: set of files for caching Thobes movements. Each directory contains cache files for different Thobe material
* **textures/**: set of images for upper and lower body color textures.
 
Rendering Variables
+++++++++++++++++++
Here we describe coding variables that control the rendering.
 
Batch Size
##########
Due to the limitation of memory on normal machines, rendering might halt after rendering several frames causing the full process to fail. For this reason, we render batch of frames in each run. **render.sh** loops to render the entire scene. To render only subset of the scene or to change the batch size, change the values of the variables in **render.sh**
 
.. figure::  docs/_static/imgs/BatchSize.png
   :width: 800px
   :height: 100px
   :align: center

Plain Character
###############
To render the plain character dataset, comment-out the **render_thobe.py** line and uncomment the **render_plain.py** in **render.sh** as in the image below

.. figure::  docs/_static/imgs/RenderPlain.png
   :width: 800px
   :height: 100px
   :align: center

Number of Lower Body Colors
###########################
As discussed above, the lower body (excluding the feet) is divided into 2, 4, or 8 colors. If you use **render.sh** for rendering, the scene will be rendered with the selected lower body descriptor in **render_thobe.py** (e.g. 4 colors). The lower body descriptor can be controlled from the following code segment in **render_thobe.py**:
 
 
.. figure::  docs/_static/imgs/LowerBodyColor.png
   :width: 800px
   :height: 200px
   :align: center

Visualize Depth Image
#####################
The depth images generated by the renderer have large integers values (1000-8000) and thus cannot be visualized by normal image software. During conversion from iff to png, depth visualization images are calculated but not saved. Depth visualization is calculated as normalized depth images with values between 0-255. In order to include depth visualization images in the rendering, uncomment the line in **iff_reader.py** where it says *...save(oDepthVisualization)*


.. figure::  docs/_static/imgs/DepthVisualization.png
   :width: 800px
   :height: 200px
   :align: center

Contact Author
++++++++++++++
| Ridwan Jalali
| jalalirsh[at]hotmail[dot]com
