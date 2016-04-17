

import sys
sys.path.append("/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python")

import maya.OpenMaya as om

import png
import numpy as np
import os
import multiprocessing as mp
from scipy.misc import toimage
from skimage import io, img_as_float
from PIL import Image, ImageChops
import shutil

ThobeColors = {
    "Background":(127, 127, 127),\
    "HeadLeftBottom":(184, 122, 0),\
    "HeadRightBottom":(73, 45, 164),\
    "HeadLeftTop":(0, 44, 105),\
    "HeadRightTop":(254, 89, 23),\
    "Nick":(213, 236, 194),\
    "LeftShoulder":(207, 197, 1),\
    "RightShoulder":(96, 101, 1),\
    "LeftTopChest":(255, 129, 0),\
    "RightTopChest":(246, 214, 255),\
    "LeftBottomChest":(1, 111, 222),\
    "RightBottomChest":(255, 207, 185),\
    "LeftUpperArm":(235, 61, 0),\
    "RightUpperArm":(112, 15, 0),\
    "LeftElbow":(143, 89, 1),\
    "RightElbow":(71, 104, 33),\
    "LeftLowerArm":(205, 90, 247),\
    "RightLowerArm":(0, 129, 160),\
    "LeftWrist":(255, 223, 138),\
    "RightWrist":(141, 41, 79),\
    "LeftHand":(118, 169, 66),\
    "RightHand":(237, 89, 141),\
    "LeftThigh":(44, 179, 37),\
    "RightThigh":(131, 116, 86),\
    "LeftKnee":(188, 148, 95),\
    "RightKnee":(168, 25, 27),\
    "LeftLeg":(142, 71, 75),\
    "RightLeg":(233, 182, 52),\
    "LeftAnkle":(65, 153, 95),\
    "RightAnkle":(85, 194, 249),\
    "LeftFoot":(0, 96, 121),\
    "RightFoot":(255, 225, 223),\
    "Thobe":(255,255,255 ),\
    "Uknown":[]\
}
BodyColors = {
    "Background":(127, 127, 127),\
    "HeadLeftBottom":(184, 122, 0),\
    "HeadRightBottom":(73, 45, 164),\
    "HeadLeftTop":(0, 44, 105),\
    "HeadRightTop":(254, 89, 23),\
    "Nick":(213, 236, 194),\
    "LeftShoulder":(207, 197, 1),\
    "RightShoulder":(96, 101, 1),\
    "LeftTopChest":(255, 129, 0),\
    "RightTopChest":(246, 214, 255),\
    "LeftBottomChest":(1, 111, 222),\
    "RightBottomChest":(255, 207, 185),\
    "LeftUpperArm":(235, 61, 0),\
    "RightUpperArm":(112, 15, 0),\
    "LeftElbow":(143, 89, 1),\
    "RightElbow":(71, 104, 33),\
    "LeftLowerArm":(205, 90, 247),\
    "RightLowerArm":(0, 129, 160),\
    "LeftWrist":(255, 223, 138),\
    "RightWrist":(141, 41, 79),\
    "LeftHand":(118, 169, 66),\
    "RightHand":(237, 89, 141),\
    "LeftThigh":(60, 154, 92),\
    "RightThigh":(131, 116, 83),\
    "LeftKnee":(188, 148, 96),\
    "RightKnee":(185, 25, 27),\
    "LeftLeg":(142, 71, 75),\
    "RightLeg":(232, 182, 51),\
    "LeftAnkle":(65, 153, 95),\
    "RightAnkle":(85, 194, 250),\
    "LeftFoot":(0, 96, 121),\
    "RightFoot":(255, 225, 223),\
    "Uknown":[]\
}
wantedColors = ThobeColors.values()
#wantedColors = BodyColors.values()


colors = []
class AsyncFactory:
    def __init__(self, func, cb_func, num_threads):
        self.func = func
        self.cb_func = cb_func
        if num_threads <> -1:
            self.pool = mp.Pool(processes=num_threads)
        else:
            self.pool = mp.Pool()
    def call(self,*args, **kwargs):
        self.pool.apply_async(self.func, args, kwargs, self.cb_func)
    
    def wait(self):
        self.pool.close()
        self.pool.join()




class MayaImage :
    """ The main class, needs to be constructed with a filename """
    def __init__(self,filename) :
        """ constructor pass in the name of the file to load (absolute file name with path) """
        # create an MImage object
        self.image=om.MImage()
        # read from file MImage should handle errors for us so no need to check
        self.image.readFromFile(filename)
        # as the MImage class is a wrapper to the C++ module we need to access data
        # as pointers, to do this use the MScritUtil helpers
        self.scriptUtilWidth = om.MScriptUtil()
        self.scriptUtilHeight = om.MScriptUtil()

        # first we create a pointer to an unsigned in for width and height
        widthPtr = self.scriptUtilWidth.asUintPtr()
        heightPtr = self.scriptUtilHeight.asUintPtr()
        # now we set the values to 0 for each
        self.scriptUtilWidth.setUint( widthPtr, 0 )
        self.scriptUtilHeight.setUint( heightPtr, 0 )
        # now we call the MImage getSize method which needs the params passed as pointers
        #as it uses a pass by reference
        self.image.getSize( widthPtr, heightPtr )
        # once we get these values we need to convert them to int so use the helpers
        self.m_width = self.scriptUtilWidth.getUint(widthPtr)
        self.m_height = self.scriptUtilHeight.getUint(heightPtr)

        # now we grab the pixel data and store
        self.charPixelPtr = self.image.pixels()
        # query to see if it's an RGB or RGBA image, this will be True or False
        self.m_hasAlpha=self.image.isRGBA()
        # if we are doing RGB we step into the image array in 3's
        # data is always packed as RGBA even if no alpha present
        self.imgStep=4
        # finally create an empty script util and a pointer to the function
        # getUcharArrayItem function for speed
        scriptUtil = om.MScriptUtil()
        self.getUcharArrayItem=scriptUtil.getUcharArrayItem

        self.scriptUtilWidth = om.MScriptUtil()
        self.scriptUtilHeight = om.MScriptUtil()

        # first we create a pointer to an unsigned in for width and height
        widthPtr = self.scriptUtilWidth.asUintPtr()
        heightPtr = self.scriptUtilHeight.asUintPtr()
    def getSize():
        self.scriptUtilWidth.setUint( widthPtr, 0 )
        self.scriptUtilHeight.setUint( heightPtr, 0 )
        self.image.getSize( widthPtr, heightPtr )
        self.m_width = self.scriptUtilWidth.getUint(widthPtr)
        self.m_height = self.scriptUtilHeight.getUint(heightPtr)
    def pixels():
        self.charPixelPtr = self.image.pixels()
    def getPixel(self,x,y) :
        """ get the pixel data at x,y and return a 3/4 tuple depending upon type """
        # check the bounds to make sure we are in the correct area
        if x<0 or x>self.m_width :
            print "error x out of bounds\n"
            return
        if y<0 or y>self.m_height :
            print "error y our of bounds\n"
            return
            # now calculate the index into the 1D array of data
        index=(y*self.m_width*4)+x*4
            # grab the pixels
        red = self.getUcharArrayItem(self.charPixelPtr,index)
        green = self.getUcharArrayItem(self.charPixelPtr,index+1)
        blue = self.getUcharArrayItem(self.charPixelPtr,index+2)
        alpha=self.getUcharArrayItem(self.charPixelPtr,index+3)
        return (red,green,blue,alpha)
    def getRGB(self,x,y) :
        r,g,b,a=self.getPixel(x,y)

        return (r,g,b)
    def width(self) :
        """ return the width of the image """
        return self.m_width
    def height(self) :
        """ return the height of the image """
        return self.m_height
    def hasAlpha(self) :
        """ return True is the image has an Alpha channel """
        return self.m_hasAlpha
    
    def getDepthImages(self):
        lPixels = self.image.depthMap()
        img_depth = np.zeros(self.width()*self.height())
        #### depths are stored as val = -1/z ; to convert multiple by -1/val ####
        ## 10 for converting to millimeter ####
        conv = -10 
        for i in xrange( 0, (self.width()*self.height()), 1 ):
            depth = om.MScriptUtil.getFloatArrayItem(lPixels, i)
            
            if depth <> 0:
                depth = conv/depth
            img_depth[i] = depth
            
        
        

        twoDimg_dept = np.flipud(np.reshape(img_depth,(self.height(),self.width())))
    
        #png.from_array(img_depth, 'L;16').save("%s/%05d_depth.png" % (folder, i))
        #png.from_array(twoDimg_dept, 'L;16').save(depthFileName)
    
        depth_visualization = visualize_depth_image(twoDimg_dept.copy())
    
        # workaround for a bug in the png module
        depth_visualization = depth_visualization.copy()  # makes in contiguous
        shape = depth_visualization.shape
        depth_visualization.shape = (shape[0], np.prod(shape[1:]))
    
        return twoDimg_dept,depth_visualization
    
    def getRGBImage(self):
        data = []
        flagged = []
        flx = 0; fly = 0
        for i in range(self.height()):
            for j in range(self.width()):
                color =self.getRGB(j,i)
                data.append(color)

                if color not in wantedColors:
                    flagged.append((i,j))
        
        rgbImage = np.reshape(data,(self.height(),self.width(),3))
  
        fixColorImage(rgbImage,flagged)
        rgbImage = np.flipud(rgbImage)
        return rgbImage

def testColorColor(aColors,color):
    if color in wantedColors:
        aColors.append(color)


def getCloserColor(im,x,y,h,w,offset=1):
    aColors= []
    if x+offset < w:
        testColorColor(aColors,tuple(im[x+offset][y]))
    if x-offset >= 0:
        testColorColor(aColors,tuple(im[x-offset][y]))
    if y+offset < h:
        testColorColor(aColors,tuple(im[x][y+offset]))
    if y-offset <= 0:
        testColorColor(aColors,tuple(im[x][y-offset]))
    if x+offset < w and y+offset < h:
        testColorColor(aColors,tuple(im[x+offset][y+offset]))
    if x+offset < w and y-offset < h:
        testColorColor(aColors,tuple(im[x+offset][y-offset]))
    if x-offset < w and y+offset < h:
        testColorColor(aColors,tuple(im[x-offset][y+offset]))
    if x-offset < w and y-offset < h:
        testColorColor(aColors,tuple(im[x-offset][y-offset]))


    if len(aColors) <= 0:
        return getCloserColor(im,x,y,h,w,offset+1)

    c = im[x][y]
    aColors = [(c[0],c[1],c[2]) for c in aColors]
    from collections import Counter
    count = Counter(aColors)
    mcs = count.most_common()

    mc = mcs[0]
    best = mc[0]
    if len(mcs) > 1:
        if mcs[1][1] == mc[1]:
            best = which_best(c,mc[0],mcs[1][0])

    return best

def which_best(c,c1,c2):
    c1s = pow(c[0]-c1[0],2)+pow(c[1]-c1[1],2)+pow(c[2]-c1[2],2)
    c2s = pow(c[0]-c2[0],2)+pow(c[1]-c2[1],2)+pow(c[2]-c2[2],2)
    if c2s > c1s:
        return c1
    return c2


def fixColorImage(im,flagged):
    for f in flagged:
        i,j = f
        rep = getCloserColor(im,i,j,len(im[0]),len(im))
        #print color, rep
        im[i][j][0] = rep[0]
        im[i][j][1] = rep[1]
        im[i][j][2] = rep[2]



def convert_to_PIL(imageArray):
    pixels = (imageArray).astype(np.uint8)
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        mode = 'RGBA'
    elif pixels.ndim == 3:
        mode = 'RGB'
    else:
        mode = 'L'
    return Image.fromarray(pixels, mode)


#im = Image.open(fileName)
        #croppedRgb = autocrop(rgbImage)
#croppedRgb.save(fileName)
def autocrop(rgb, bgcolor=(127,127,127)):
    if rgb.mode != "RGB":
        rgb = rgb.convert("RGB")
    bg = Image.new("RGB", rgb.size, bgcolor)
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    return bbox



def visualize_depth_image(data):
    
    data[data == 0.0] = np.nan
    
    maxdepth = np.nanmax(data)
    mindepth = np.nanmin(data)
    data = data.copy()
    data -= mindepth
    data /= (maxdepth - mindepth)
    
    gray = np.zeros(list(data.shape) + [3], dtype=data.dtype)
    data = (1.0 - data)
    gray[..., :3] = np.dstack((data, data, data))
    
    # use a greenish color to visualize missing depth
    gray[np.isnan(data), :] = (97, 160, 123)
    gray[np.isnan(data), :] /= 255
    
    #gray = exposure.equalize_hist(gray)
    
    # set alpha channel
    gray = np.dstack((gray, np.ones(data.shape[:2])))
    gray[np.isnan(data), -1] = 0.5
    
    return gray * 255


def get_output_names(outName):
    #fileName = os.path.basename(iffImageFile).split("_")[-1].replace(".iff",".png")
    return outName+"_ground_truth.png",outName+"_depth.png",outName+"_depth_visualization.png"#.replace(".png","_ground_truth.png"),fileName.replace(".png","_depth.png"),fileName.replace(".png","_depth_visualization.png")

def convert_image(iffImageFile,outName):
    print iffImageFile
    img=MayaImage(iffImageFile)
    oGTruth, oDepth, oDepthVisualization = get_output_names(outName)

    dImage, dvImage = img.getDepthImages()
    rgbImage = img.getRGBImage()
    x1,x2,y1,y2 = autocrop(convert_to_PIL(rgbImage))
    
    oRgb = rgbImage
    oRgb = oRgb[x2:y2,:]
    oRgb = oRgb[:,x1:y1]
    oRgb = np.reshape(oRgb,(y2-x2,(y1-x1)*3))

    png.from_array(oRgb, 'RGB;8').save(oGTruth)

    
    oDvImage = dvImage
    oDvImage = np.reshape(oDvImage,(img.height(),img.width(),4))
    oDvImage = oDvImage[x2:y2,:]
    oDvImage = oDvImage[:,x1:y1]
    oDvImage = np.reshape(oDvImage,(y2-x2,(y1-x1)*4))
    #png.from_array(oDvImage, 'RGBA;8').save(oDepthVisualization)
    
    
    oDImage = np.reshape(dImage,(img.height(),img.width(),1))
    oDImage = oDImage[x2:y2,:]
    oDImage = oDImage[:,x1:y1]
    oDvImage = np.reshape(oDImage,(y2-x2,(y1-x1)))
    
    png.from_array(oDvImage, 'L;16').save(oDepth)
    #shutil.copy(oGTruth, oGTruth.replace("ground_truth","colors"))
    return iffImageFile
import fnmatch
def get_iff_files(iffPath,pattern=None):
    if pattern is None:
        pattern = "*.iff"
    else:
        pattern = "*%s.iff" % (pattern)
    
    matches = []
    for root, dirnames, filenames in os.walk(iffPath):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches

def print_finished_image(i):
    print "Finished proccessing:",i


def convert(path, prefix=None, numThreads=-1, offset=1, parallel=False,only=None):
    if prefix == None: prefix = path


    if os.path.isfile(path):
        convert_image(path,prefix)

    if os.path.isdir(path):
        ins = get_iff_files(path,only)
        outs = range(offset,offset+len(ins))
        inout = zip(ins,outs)


        if not parallel:
            for inf,outf in inout:
                num = '%05d' % outf
                convert_image(inf,"%s/%s" %(prefix,num))
        else:

            print "converting with ",numThreads, "threads"
            async_convert = AsyncFactory(convert_image, print_finished_image,numThreads) 
            for inf,outf in inout:
                num = '%05d' % outf
                async_convert.call(inf,"%s/%s" % (prefix,num))
            async_convert.wait()



if __name__ == "__main__":
    
    if len(sys.argv)< 2:
        print("usage: %s <iff images folder> [<num_threads>]" % sys.argv[0])
        sys.exit(0)
    iffPath = sys.argv[1]


    outPath = sys.argv[2]


    convert(iffPath,outPath)

    
    
