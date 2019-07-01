from Dataset import Dataset
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Input
import numpy as np
import pickle
import argparse

import NeuMF
import GMF
import MLP

def parse_args():
	parser = argparse.ArgumentParser(description="Run Demo.")
	parser.add_argument('--t', default='[]',help='Input type.')
	parser.add_argument('--r', default='[]',help='Input region.')
	parser.add_argument('--m', default='ai_NeuMF_64_[64,32,16,8]_1560254024.h5',help='Input model.')
	return parser.parse_args()


def Diff(listA, listB):
	if len(listA) != len(listB):
		return -1

	diff = 0.0
	for i in range(len(listA)):
		diff += (listA[i] - listB[i]) ** 2
	
	return diff ** 0.5


# Open all file
with open('Data/ai.train_data.pkl','r') as f:
    data = pickle.load(f)

with open('Data/ai.train_data.pkl', 'r') as f:
	trainData = pickle.load(f)

with open('Data/attractionToindex.pkl', 'r') as f:
	attr2index = pickle.load(f)

with open('Data/ai.user_type_prob.pkl', 'r') as f:
	dataType = pickle.load(f)

with open('Data/type_attraction_dic.pkl', 'r') as f:
	typeAttraction = pickle.load(f)

with open('Data/type_index.pkl', 'r') as f:
	type2index = pickle.load(f)

with open('Data/region_attraction_dic.pkl', 'r') as f:
	region = pickle.load(f)

args = parse_args()

premodel = args.m


dataset = Dataset('Data/ai')


train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
num_users, num_items = train.shape
num_types = len(type2index)

#print("user : ", num_users, "items", num_items)
#print("train = ", train, "test = ", testRatings, "Neg = ", testNegatives)

# Save as weight
model = NeuMF.get_model(num_users, num_items, 64, [64, 32, 16, 8], [0,0,0,0], 0.0)

# model = GMF.get_model(num_users, num_items, 8)

model.load_weights('/shared_home/chuying/AI/NCF/Pretrain/' + premodel)


attr = [""] * num_items
for a, i in attr2index.iteritems():
	attr[i] = a
# attr = list(attr2index.keys())
# typ = list(type2index.keys())
UserType = [0.0] * len(dataType[0])


#for i in range(len(type2index)):
#	print(i, ", ", type2index[i])
#print(len(attr), len(dataType[0]))

#############
# User input
iWant = eval(args.t)
iWant_R = eval(args.r)
#############
if len(iWant) == 0:
	score = 1.0
else:
	score = 1 / len(iWant)
for i in range(len(iWant)):
	UserType[i] = score

# Find the most close user in dataset
minDiff = float("inf")
targetUser = 0
for i in range(num_users):
	tmp = Diff(UserType, dataType[i]) 
	if tmp < minDiff:
		minDiff = tmp
		targetUser = i

# print(targetUser, minDiff)

############
# Model predict
user = np.full(num_items, targetUser, dtype = 'int32')
item = np.arange(num_items)
types = []
for i in range(len(item)):
	for typ, a in typeAttraction.iteritems():
		if attr[item[i]] in a:
			types.append(type2index.index(typ))

result = model.predict([user, item])

result = result.flatten()

# Sorted attractions
k_index = result.argsort()[::-1]




# Print data
for u, attr_index in trainData:
	if u == targetUser:
		tmp = attr[attr_index]
		print(tmp)
		for t, att in typeAttraction.iteritems():
			if tmp in att:
				print(t)
				print("-----")
				break


iWantType = []
if len(iWant) != 0:
	print("==========")
	print("Input Type : ")
for i in range(len(iWant)):
	tmp = type2index[iWant[i]]
	iWantType.append(tmp)
	print(tmp)


regionIndex = list(region.keys())
iWantRegion = []
if len(iWant_R) != 0:
	print("=========")
	print("Input Region : ")
for i in range(len(iWant_R)):
	tmp = regionIndex[iWant_R[i]]
	iWantRegion.append(tmp)
	print(tmp)



def RA_ByType(k_index, top_k, attr, iWantType, typeAttraction):		
	count = 0
	RA = []
	for i in range(len(k_index)):
		if top_k < count:
			break
		tmp = attr[k_index[i]]
		for j in range(len(iWantType)):
			if tmp in typeAttraction[iWantType[j]] and tmp not in RA:
				count += 1
				RA.append(tmp)
				print(tmp)
	return RA

def RA_ByTypeAndRegion(k_index, top_k, attr, iWantType, iWantRegion, typeAttraction, region):
	count = 0
	RA = []
	for i in range(len(k_index)):
		if top_k < count:
			break
		tmp = attr[k_index[i]]
		for x in range(len(iWantRegion)):
			if tmp in region[iWantRegion[x]] and tmp not in RA:
				for y in range(len(iWantType)):
					if tmp in typeAttraction[iWantType[y]]:
						count += 1
						RA.append(tmp)
						print(tmp)
	return RA


print("==========")
print("Recommend Attractions : ")

if len(iWant) != 0 and len(iWant_R) != 0:
	test = RA_ByTypeAndRegion(k_index, 10, attr, iWantType, iWantRegion, typeAttraction, region)
elif len(iWant) != 0:
	RA_Type = RA_ByType(k_index, 10, attr, iWantType, typeAttraction)
elif len(iWant_R) != 0:
	test = RA_ByType(k_index, 10, attr, iWantRegion, region)



# Without restricting dataset
'''
for i in range(10):
	print(attr[k_index[i]])
'''






'''
print("Index : ")
print(k_index)
print("Probability :")
print(result[k_index])
'''



