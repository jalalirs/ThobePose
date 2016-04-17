from PIL import Image
import sys
import os


def combine(files,oName="4colors_pred.png"):
	grid = 6
	size = 200
	files = files[:grid*grid]
	#creates a new empty image, RGB mode, and size 400 by 400.
	new_im = Image.new('RGBA', (size*grid,size*grid))

	for i,f in enumerate(files):

		im = Image.open(f)
		#Here I resize my opened image, so it is no bigger than 100,100
		im.thumbnail((size,size))
		x = i%grid*size
		y = i/grid*size

		#paste the image at location i,j:
		new_im.paste(im, (x,y),im)
	
	new_im.save(oName)





def get_png_files(path):
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if "prediction.png" in f or "ground" in f]



if __name__ == "__main__":
    
	if len(sys.argv)< 2:
		print("usage: %s <png images> " % sys.argv[0])
		sys.exit(0)
	filesPath = sys.argv[1]

	files = get_png_files(filesPath)

	combine(files)
	
