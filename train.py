import os
import ast
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain
import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE = "cuda"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HIIGHT = int(os.environ.get("IMG_HIIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = os.environ.get(os.environ.get("MODEL_MEAN"))
MODEL_STD = os.environ.get(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = int(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = int(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get(os.environ.get("BASE_MODEL"))

def loss_fn(outputs, torgets):
	o1, o2, o3 = outputs
	t1, t2, t3 = torgets
	l1 = nn.CrossEntropyLoss()(o1,t1)
	l2 = nn.CrossEntropyLoss()(o2,t2)
	l3 = nn.CrossEntropyLoss()(o3,t3)
	return (l1 + l2 + l3)/3

def train(dataset, data_loader, model, optimizer):
	model.train()
	for batch_index, data in tqdm(enumerate(data_loader), total=len(dataset)/data_loader.batch_size) :
		image = data["image"]
		grapheme_root = data["grapheme_root"]
		vowel_diacritic = data["vowel_diacritic"]
		consonant_diacritic = data["consonant_diacritic"]
		image = image.to(DEVICE, dtype=torch.float)
		grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
		vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
		consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)
		optimizer.zero_grad()
		outputs = model(image)
		torgets = (grapheme_root, vowel_diacritic, consonant_diacritic)
		loss = loss_fn(outputs, torgets)
		
		loss.backword()
		optimizer.step()
		
def evaluate(dataset, data_loader, model):
	model.eval()
	final_loss = 0
	counter = 0
	for batch_index, data in tqdm(enumerate(data_loader), total=len(dataset)/data_loader.batch_size) :
		counter = counter + 1
		image = data["image"]
		grapheme_root = data["grapheme_root"]
		vowel_diacritic = data["vowel_diacritic"]
		consonant_diacritic = data["consonant_diacritic"]
		
		image = image.to(DEVICE, dtype=torch.float)
		grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
		vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
		consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)
		
		outputs = model(image)
		torgets = (grapheme_root, vowel_diacritic, consonant_diacritic)
		loss = loss_fn(outputs, torgets)
		
		final_loss += loss
	reurn final_loss/counter
	


def main():
	model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
	model.to(DEVICE)
	
	train_dataset = BengaliDatasetTrain(
		folds = TRAINING_FOLDS,
		img_height = IMG_HIIGHT,
		img_width = IMG_WIDTH,
		mean = MODEL_MEAN,
		std = MODEL_STD
	)	
	
	train_loader = torch.utils.data.dataLoader(
		dataset = train_dataset,
		batch_size = TRAINING_BATCH_SIZE,
		shuffle = True,
		num_workers = 4
	)
	
	valid_dataset = BengaliDatasetTrain(
		folds = VALIDATION_FOLDS,
		img_height = IMG_HIIGHT,
		img_width = IMG_WIDTH,
		mean = MODEL_MEAN,
		std = MODEL_STD
	)	
	
	valid_loader = torch.utils.data.dataLoader(
		dataset = valid_dataset,
		batch_size = TEST_BATCH_SIZE,
		shuffle = False,
		num_workers = 4
	)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.3, verbose=True)
	
	if torch.cuda.device_count()>1:
		model = nn.dataParallel(model)
		
	for epoch in range(EPOCHS):
		train(train_dataset, train_loader, model, optimizer)
		val_score = evaluate(train_dataset, train_loader, model)
		scheduler.step(val_score)
		torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")
		
if __name__ = "__main__":
	main()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	