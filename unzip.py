import os
import shutil

data_dir = "data"
if not os.path.exists(data_dir):
	zip_dir = data_dir + ".zip"
	print("unzipping NBA data...")
	shutil.unpack_archive(zip_dir)