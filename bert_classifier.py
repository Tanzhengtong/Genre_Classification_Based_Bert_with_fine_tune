# -*- coding: utf-8 -*-

!pip install transformers
import nltk
nltk.download('stopwords')
import json
import numpy as np
import pandas as pd
import time
import random
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import roc_auc_score,f1_score

from torchtext.legacy import data
import spacy
import datetime

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertModel
from transformers import BertTokenizer
from imblearn.over_sampling import RandomOverSampler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def remove_punctuation(line):
    rule = re.compile(r"[^a-zA-Z0-9]") #remove all the non-characters
    line = rule.sub('',line)
    return line

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# Data check
print("Total number of samples is "+str(len(X)))
max_len = 0
ind = [100,200,400,512]
for i in ind:
  count = 0
  for x in X:
      max_len = max(max_len, len(x))
      if len(x)>i:
        count+=1
  print("The number of sentence length over {} is: ".format(i), count)
print('Max sentence length: ', max_len)

# check unbalanced data set
value_cnt = {} # dic store frequency
for value in Y:
	value_cnt[value] = value_cnt.get(value, 0) + 1
print(value_cnt)
# showing unbalanced dataset from the result
# oversampling
X=np.array(X)
Y=np.array(Y)
X=np.expand_dims(X, axis=1)
over = RandomOverSampler(sampling_strategy='all')#(sampling_strategy=1)
X, Y = over.fit_resample(X, Y)
X=np.squeeze(X)

# Tokenization
# clean data
content_clean = []
# clear common words
stop_words=set(stopwords.words('english'))
for s in X:
    s=s.lower()
    s=[word for word in s.split()]
    text_clean =""
    for i in s:
        i=remove_punctuation(i)
        # words that only contain characters
        if len(i)>=1 and i not in stop_words:
            text_clean+=i
            text_clean+=" "
    content_clean.append(text_clean)
print("There are "+ str(len(content_clean))+" Clean data ")

# Tokenization 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
input_ids = []
attention_masks = []
for content in content_clean:
    temp_dict = tokenizer.encode_plus(
                        content,                      # Sentence to encode.
                        add_special_tokens = True, # Add Classification
                        max_length = 512,           # The length fit the bert model
                        pad_to_max_length = True,
                        truncation=True,
                        return_attention_mask = True,  # add attention mask
                        return_tensors = 'pt',    
                   )
    input_ids.append(temp_dict['input_ids'])
    attention_masks.append(temp_dict['attention_mask'])
# Convert the format
# step 1
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(Y)
# step 2
dataset = TensorDataset(input_ids, attention_masks, labels)
# Split the data into training set and validation set
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=torch.Generator().manual_seed(27))

# Batch normalization
batch_size =16
training_dataloader = DataLoader(
            train_dataset,  
            shuffle = True,
            batch_size = batch_size 
        )
validation_dataloader = DataLoader(
            val_dataset,
            shuffle = False,
            batch_size = batch_size 
        )

# Model definition
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 4, 
    output_attentions = False, 
    output_hidden_states = False,
)
model.to(device)
torch.cuda.empty_cache()

# Hyperparameter setting
optimizer = AdamW(model.parameters(),
                  lr = 6e-5, 
                  eps = 1e-8,
                  weight_decay =1e-2,
                )
epochs = 4
criterion = nn.CrossEntropyLoss()
seed_val = 27
random.seed(seed_val)
torch.manual_seed(seed_val)

for epoch in range(0, epochs):
    print('Epoch: '+str(epoch+1))
    print("Training")
    train_loss = 0
    train_accuracy = 0
    model.train()
    for step, batch in enumerate(training_dataloader):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        model.zero_grad()        
        out = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        # accuracy
        temp = out[1]
        pred = torch.argmax(temp, dim = 1)
        train_accuracy +=  torch.sum(pred == labels).item()

        # loss
        loss = out[0]
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    avg_train_accuracy = train_accuracy / len(training_dataloader.dataset)
    avg_train_loss = train_loss / len(training_dataloader.dataset)
    print("Accuracy "+ str(avg_train_accuracy))
    print("Training loss "+ str(avg_train_loss))
    
    # Validation
    print("Validation")
    model.eval()
    eval_accuracy = 0
    eval_loss = 0
    y_true = []
    y_pred = []

    for batch in validation_dataloader:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():        
            out = model(input_ids, token_type_ids=None, attention_mask=input_mask,labels=labels)
            loss = out[0]
            temp = out[1] 
        # loss
        eval_loss += loss.item()
        pred = torch.argmax(temp, dim = 1)
        # accuracy
        eval_accuracy += torch.sum(pred == labels).item()
        y_true.append(labels.flatten())
        y_pred.append(pred.flatten())
        
    avg_val_accuracy = eval_accuracy / len(validation_dataloader.dataset)
    print("Accuracy "+ str(avg_val_accuracy))
    avg_val_loss = eval_loss / len(validation_dataloader.dataset)
    print("Validation loss "+ str(avg_val_loss))
   
    y_true = torch.cat(y_true).tolist()
    y_pred = torch.cat(y_pred).tolist()
    f1 = f1_score( y_true, y_pred, average='macro' )
    print('macro averaged F1 score: ',f1)
    print()

# Load the test data
test_data = json.load(open("/content/drive/MyDrive/Colab Notebooks/genre_test.json", "r"))
Xt = test_data['X']

# tokenization
content_clean_test = []
# clear common words
stop_words=set(stopwords.words('english'))
for s in Xt:
    s=s.lower()
    s=[word for word in s.split()]
    text_clean =""
    for i in s:
        i=remove_punctuation(i)
        # words that only contain alphanumeric characters
        if len(i)>=1 and i not in stop_words:
            text_clean+=i
            text_clean+=" "
    content_clean_test.append(text_clean)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
input_ids_test = []
attention_masks_test = []
for tweet in content_clean_test:
    temp_dict = tokenizer.encode_plus(
                        tweet,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        truncation=True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    input_ids_test.append(temp_dict['input_ids'])
    attention_masks_test.append(temp_dict['attention_mask'])
# Convert the format
input_ids_test = torch.cat(input_ids_test, dim=0)
attention_masks_test = torch.cat(attention_masks_test, dim=0)
dataset = TensorDataset(input_ids_test, attention_masks_test)
batch_size = 256
test_dataloader = DataLoader(
            dataset,  
            shuffle = False,
            batch_size = batch_size)

# Make prediction
model.eval()
result=np.array
for step, batch in enumerate(test_dataloader):
  with torch.no_grad():
     id=batch[0].to(device)
     mask=batch[1].to(device)
     raw_pred = model(id,mask)
  logits = raw_pred[0]
  logits = logits.detach().cpu().numpy()
  pred = np.argmax(logits, axis = 1)
  # print(step)
  result=np.concatenate((result, pred), axis=None)

test_result=result[1:].tolist()
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(test_result): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()
