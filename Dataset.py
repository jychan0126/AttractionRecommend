'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pickle

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train_data.pkl")
        self.testRatings = self.load_rating_file_as_list(path + ".test_data.pkl")
        self.testNegatives = self.load_negative_file(path + ".test_neg.pkl")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        '''
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        '''
        with open(filename, "r") as f:
            data = pickle.load(f)
            for i in range(len(data)):
                ratingList.append([data[i][0], data[i][1]])
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        '''
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        '''
        with open(filename, "r") as f:
            data = pickle.load(f)
            for i in range(len(data)):
                neg = []
                for x in range(50):
                    neg.append(data[i][x])
                negativeList.append(neg)
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        #filename = 'Data/train_datas_p2.pkl'
        # Get number of users and items
        num_users, num_items = 0, 0
        users, items = [], []
		
        with open(filename, "r") as f:
            data = pickle.load(f)
            for i in range(len(data)):
                #users.append(data[i][0])
                #items.append(data[i][1])

                num_users = max(num_users, data[i][0])
                num_items = max(num_items, data[i][1])
            # print(num_users, num_items)
            #print(np.unique(users).shape, np.unique(items).shape)
        
        # Open relative file
        with open('Data/attractionToindex.pkl', "r") as f:
            itemIndex = pickle.load(f)
            # {attraction : index}

        with open('Data/ai.user_type_smax.pkl', "r") as f:
            userType = pickle.load(f)

        with open('Data/type_attraction_dic.pkl', "r") as f:
            itemType = pickle.load(f)
            # {type : attr1, attr2, ...}

        with open('Data/type_index.pkl', 'r') as f:
            type2index = pickle.load(f)
            # [attr1, attr2, attr3, ...]

        # Construct matrix`
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            data = pickle.load(f)
            '''
            for i in range(len(data)):
                user, item = int(data[i][0]), int(data[i][1])
                mat[user, item] = 1.0

            '''
            for i in range(len(data)):
                user, item = int(data[i][0]), int(data[i][1])
                for attr, index in itemIndex.iteritems():
                    if index == item:
                        attrName = attr
                        break

                for typ, attr in itemType.iteritems():
                    if attrName in attr:
                        attrTypeName = typ

                # type2index = list(itemType.keys())
                attrType = type2index.index(attrTypeName)
                #print(userType[user][attrType])
                #uType, iType = userType[user], itemType[item]
                mat[user, item] = userType[user][attrType]
           
        return mat
