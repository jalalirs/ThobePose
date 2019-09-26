from PIL import Image
import png
import sys
import numpy as np
import os



def get_depth_files(path):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if "depth.png" in f ]

def convert_to_PIL(imageArray,mode="RGB"):
    pixels = (imageArray).astype(np.uint8)
    return Image.fromarray(pixels, mode)

def compress(files,binFileName):
	images = []
	print "Reading images...",
	for f in files:
		prediction = f.replace("depth.png","ground_truth.png")
		depth = np.asarray(Image.open(f))
		pred = np.asarray(Image.open(prediction))
		images.append(depth)
		images.append(pred)
	print "\t\t done."
	images = np.array(images)
	np.save(binFileName,images)

def uncompress(binFileName,outputDir):
	images = np.load(binFileName)
	for i in range(0,len(images),2):
		imageId = i/2
		depth = images[i]
		prediction = images[i+1]
		prediction = prediction.reshape(len(prediction),len(prediction[0])*3)
		png.from_array(depth, 'L;16').save("%s/%d_depth.png" % (outputDir,imageId))
		png.from_array(prediction, 'RGB;8').save("%s/%d_ground_truth.png" % (outputDir,imageId))

if __name__ == "__main__":
    
	if len(sys.argv)< 4:
		print("usage: %s <command {compress | uncompress}> <files path (compress) or binary file (uncompress)> <output file (compress) output dir (uncompress)>" % sys.argv[0])
		sys.exit(0)

	command = sys.argv[1]

	inPath = sys.argv[2]
	outPath = sys.argv[3]
	if command == "uncompress":
		uncompress(inPath,outPath)
	elif command == "compress":
		files = get_depth_files(inPath)
		compress(files,outPath)
	else:
		print("usage: %s <command {compress | uncompress}> <files path (compress) or binary file (uncompress)> <output file (compress) output dir (uncompress)>" % sys.argv[0])
		sys.exit(0)
	


	
	

	
	




		
	
