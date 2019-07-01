# -*- coding: utf-8 -*
import h5py
import numpy as np
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
'''
import pickle


filename = 'Data/type_attraction_dic.pkl'

with open(filename, "r") as f:
	#byte = f.read()
	data = pickle.load(f)

with open("Data/attractionToindex.pkl", 'r') as f:
	itemIndex = pickle.load(f)

with open('Data/region_attraction_dic.pkl') as f:
	regionAttr = pickle.load(f)

with open('Data/type_list.pkl', 'r') as f:
	typ = pickle.load(f)

for i in range(len(typ)):
	print(typ[i])
'''
inputItem = 12
for attr, index in itemIndex.iteritems():
	if index == inputItem:
		goal = attr
print(goal)

for typ, attr in data.iteritems():
	if goal in attr:
		res = typ

print(res)
'''
