import pandas as pd
import numpy as np
import joblib
import glob
from tqdm import tqdm

if name == "__main__"
	files = glob.glob(../imput/train*.parquet)
	for f in files:
		df = pd.read_parquet(f)
		image_ids = df.image_id.values
		df = df.drop("image_id", axiz=1)
		image_array = df.values
		for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids))):
			joblib.dump(image_array[j,:],f"../imput/image_pickles/{img_id}.pkl")