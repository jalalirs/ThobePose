from PIL import Image
import png
import sys
import numpy as np
import os


def get_png_files(path):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if "prediction.png" in f ]


def make_transparent(f):
	img = Image.open(f)
	img = img.convert("RGBA")
	datas = img.getdata()

	newData = []
	for i,item in enumerate(datas):
		65,153,95
		if item[0] == 65 and item[1] == 153 and item[2] == 95:
			print f
			exit()
		#	newData.append((255, 255, 255, 0))
		#else:
		#	newData.append(item)

	img.putdata(newData)
	img.save(f, "PNG")

if __name__ == "__main__":
    
	if len(sys.argv)< 2:
		print("usage: %s <png images> " % sys.argv[0])
		sys.exit(0)
	filesPath = sys.argv[1]
	if len(filesPath.split("/")[-1].split(".")) > 1:
		files = [filesPath]
	else:
		files = get_png_files(filesPath)
	for f in files:

		groundt = f.replace("prediction.png","ground_truth.png")
		print f
		make_transparent(f)
		make_transparent(groundt)
	
