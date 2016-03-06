import os 
from PIL import Image
from resizeimage import resizeimage

path_cor = '../../data/COR/'
dest_path_cor = '../../data/COR_44/'
size_cor = [44, 44]
name_cor = "_44_44"

path_sag = '../../data/SAG/'
dest_path_sag = '../../data/SAG_44/'
size_sag = [52, 44]
name_sag = "_52_44"

path_tra = '../../data/TRA/'
dest_path_tra = '../../data/TRA_44/'
size_tra = [44, 52]
name_tra = "_44_52"

def resize(path, dest_path, size, size_name):
	for fn in os.listdir(path):
		name = fn[:-5]
		with open(path+fn, 'r+b') as f:
			with Image.open(f) as im:
				cover = resizeimage.resize_cover(im, size)
				cover.save(dest_path+name+size_name, im.format)

resize(path_cor, dest_path_cor, size_cor, name_cor)
resize(path_sag, dest_path_sag, size_sag, name_sag)
resize(path_tra, dest_path_tra, size_tra, name_tra)
