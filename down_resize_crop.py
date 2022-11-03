import wget
from PIL import Image
import os
import tarfile
import shutil
cwd = os.getcwd()

handbag_url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2handbags.tar.gz'
shoes_url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz'

wget.download(handbag_url)
wget.download(shoes_url)

with tarfile.open('./edges2handbags.tar.gz') as tar:
def is_within_directory(directory, target):
	
	abs_directory = os.path.abspath(directory)
	abs_target = os.path.abspath(target)

	prefix = os.path.commonprefix([abs_directory, abs_target])
	
	return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

	for member in tar.getmembers():
		member_path = os.path.join(path, member.name)
		if not is_within_directory(path, member_path):
			raise Exception("Attempted Path Traversal in Tar File")

	tar.extractall(path, members, numeric_owner=numeric_owner) 
	

safe_extract(tar)
	tar.close()

with tarfile.open('./edges2shoes.tar.gz') as tar:
def is_within_directory(directory, target):
	
	abs_directory = os.path.abspath(directory)
	abs_target = os.path.abspath(target)

	prefix = os.path.commonprefix([abs_directory, abs_target])
	
	return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

	for member in tar.getmembers():
		member_path = os.path.join(path, member.name)
		if not is_within_directory(path, member_path):
			raise Exception("Attempted Path Traversal in Tar File")

	tar.extractall(path, members, numeric_owner=numeric_owner) 
	

safe_extract(tar)
	tar.close()

bag_dir = "/edges2handbags/train/"
shoe_dir = "/edges2shoes/train/"

bag_list = os.listdir("."+bag_dir)
shoe_list = os.listdir("."+shoe_dir)

os.mkdir("./bags")
os.mkdir("./shoes")
os.mkdir("./result")

for i in bag_list:
	image = Image.open(cwd+bag_dir+i)
	image = image.resize([128,64])
	image = image.crop([64,0,128,64])
	image.save("./bags/"+i)

for i in shoe_list:
	image = Image.open(cwd+shoe_dir+i)
	image = image.resize([128,64])
	image = image.crop([64,0,128,64])
	image.save("./shoes/"+i)

#shutil.rmtree("./edges2handbags")
#shutil.rmtree("./edges2shoes")

os.remove('./edges2handbags.tar.gz')
os.remove('./edges2shoes.tar.gz')
