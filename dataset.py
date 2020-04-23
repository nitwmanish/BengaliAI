import pandas as pd
import numpy as np
import joblib
from PIL import Image
import albumentations
import torch

class BengaliDatasetTrain:
	def __init__(self, folds, img_height, img_width, mean, std)
		df = pd.read_csv("../imput/train_folds.csv")
		df = df[["image_id","grapheme_root","vowel_diacritic","consonant_diacritic", "kfold"]]
		df = df[df.kfold.isin[folds]].reset_index(drop=True)
		self.image_ids = df.image_id.values
		seld.grapheme_root = df.grapheme_root.values
		self.vowel_diacritic = df.vowel_diacritic.values
		self.consonant_diacritic = df.consonant_diacritic.values
		if len(folds) == 1:
			self.aug = albumentations.Compose([
				albumentations.Resize(img_height,img_width, always_apply=True),
				albumentations.Normalize(mean, std, always_apply=True)
			])
		else :
			self.aug = albumentations.Compose([
				albumentations.Resize(img_height,img_width, always_apply=True),
				albumentations.ShiftScaleRotate(shift_limit=0.0625,
				scale_limit=0.1, rotate_limit=5,
				p=0.9),
				albumentations.Normalize(mean, std, always_apply=True)
			])
		
	def __len__(self):
		return len(self.image_ids)
		
	def __getitem__(self, item):
		image = joblib.load(f"../input/image_pickles/{self.image_ids[item]}.pkl")
		image = image.reshape(137, 236).astype(float)
		image = Image.fromarray(image).convert("RGB")
		image = self.aug(image=np.array(image))["image"]
		image = np.transpose(image, (2, 0, 1)).astype(np.float32)
		return {
			'image':torch.tensor(image, dtype=torch.float),
			'grapheme_root':torch.tensor(self.grapheme_root[item], dtype=torch.long),
			'vowel_diacritic':torch.tensor(self.vowel_diacritic[item], dtype=torch.long),
			'consonant_diacritic':torch.tensor(self.consonant_diacritic[item], dtype=torch.long)		
		}
		