import sys
sys.path.append("/Applications/Autodesk/maya2015/Maya.app/Contents/Frameworks/Python.framework/Versions/Current/lib/python2.7/site-packages/numpy-1.9.1-py2.7-macosx-10.6-intel.egg")
import maya.standalone
maya.standalone.initialize("Python")




import maya.cmds as cmds
import numpy as np
import math as ma
import gc


import maya.utils as utils
import threading
import time

cmds.setAttr("hardwareRenderGlobals.graphicsHardwareGeometryCachingData", 0)
cmds.setAttr("hardwareRenderGlobals.maximumGeometryCacheSize", 2)


MAYA_FILE = "../scenes/Hamed.mb"
cmds.file(MAYA_FILE, force=True, open=True)





def main():
    global outPath
    global inPath
    if len(sys.argv)< 2:
        print("usage: %s <png images folder> <output path> <type: d for depth, dv for depth visualization, g for ground truth>[<num_threads>]" % sys.argv[0])
        sys.exit(0)
    start = int(sys.argv[1])
    end = int(sys.argv[2])




    for i in range(start,end):
            cmds.currentTime(i)
            start_time = time.time()
            cmds.hwRender(currentFrame= True)
            iff.convert(iffsDir,pngDir,only=str(i),offset=16*i)
            print("finished timeframe %d --- %s seconds ---" % (i,time.time() - start_time))
            cmds.flushUndo()
            cmds.clearCache( all=True )
            cmds.DeleteHistory()




if __name__ == "__main__":
    main()



    



