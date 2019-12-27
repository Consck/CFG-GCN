import numpy as np
import scipy.sparse as sp
import torch
import random

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

#源代码
def load_data(path="/home/lab106/bjy/pygcn/data/cora/", dataset="pca_cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    
    #resultList=random.sample(range(1,1434),50)    
    #print("resultList = {}".format(resultList))
    #resultList = [68, 1051, 1170, 1266, 431, 509, 591, 710, 46, 488, 1053, 53, 1160, 1307, 1395, 623, 1335, 828, 759, 45, 101, 883, 1211, 857, 1199, 128, 279, 1331, 191, 511, 233, 636, 99, 1273, 737, 580, 452, 764, 1419, 773, 1161, 1426, 100, 965, 950, 990, 845, 1424, 7, 493]
    #resultList=range(1,200)
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))    
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    '''
    idx_train = range(199)
    idx_val = range(200, 500)
    idx_test = range(500,1500)
    '''
    idx_train = random.sample(range(1,2708),1083)
    idx_val = random.sample(range(1,2708),300)
    idx_test1 = range(1,2708)
    idx_test2 = list(set(idx_train).union(set(idx_val)))
    idx_test = list(set(idx_test1) ^ set(idx_test2))
    
    print(len(idx_train))
    print(len(idx_val))
    print(len(idx_test))
    
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
    
#mydata读取代码
def new_load_data(path="/home/lab106/bjy/pygcn/data/new_cora/", dataset="new_cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))    
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    
    #取出全部数据进行测试
    #idx_train = range(1,198)
    #idx_val = range(1,198)
    #idx_test = range(1,197)
    #1-9-21
    #idx_train=random.sample(range(1,16),3)+random.sample(range(17,43),3)+random.sample(range(44,67),3)+random.sample(range(68,92),3)+random.sample(range(93,140),3)+random.sample(range(141,185),3)+random.sample(range(186,197),3)  
    #idx_val=random.sample(range(1,16),1)+random.sample(range(17,43),1)+random.sample(range(44,67),2)+random.sample(range(68,92),2)+random.sample(range(93,140),3)+random.sample(range(141,185),3)+random.sample(range(186,197),2)
    
    #25
    #idx_train=random.sample(range(1,16),3)+random.sample(range(17,43),3)+random.sample(range(44,67),5)+random.sample(range(68,92),5)+random.sample(range(93,140),3)+random.sample(range(141,185),3)+random.sample(range(186,197),3)  
    #idx_val=random.sample(range(1,16),1)+random.sample(range(17,43),2)+random.sample(range(44,67),3)+random.sample(range(68,92),3)+random.sample(range(93,140),2)+random.sample(range(141,185),3)+random.sample(range(186,197),2)
    
    #31
    #idx_train=random.sample(range(1,16),3)+random.sample(range(17,43),5)+random.sample(range(44,67),5)+random.sample(range(68,92),5)+random.sample(range(93,140),5)+random.sample(range(141,185),5)+random.sample(range(186,197),3)  
    #idx_val=random.sample(range(1,16),1)+random.sample(range(17,43),2)+random.sample(range(44,67),3)+random.sample(range(68,92),3)+random.sample(range(93,140),2)+random.sample(range(141,185),3)+random.sample(range(186,197),2)
    
    #2-8-40
    #idx_train=random.sample(range(1,16),3)+random.sample(range(17,43),5)+random.sample(range(44,67),5)+random.sample(range(68,92),5)+random.sample(range(93,140),10)+random.sample(range(141,185),9)+random.sample(range(186,197),3) 
    #idx_val=random.sample(range(1,16),3)+random.sample(range(17,43),5)+random.sample(range(44,67),5)+random.sample(range(68,92),5)+random.sample(range(93,140),10)+random.sample(range(141,185),5)+random.sample(range(186,197),3)
    
    #3-7
    #idx_train=random.sample(range(1,16),5)+random.sample(range(17,43),8)+random.sample(range(44,67),7)+random.sample(range(68,92),7)+random.sample(range(93,140),14)+random.sample(range(141,185),13)+random.sample(range(186,197),3)
    #idx_val=random.sample(range(1,16),5)+random.sample(range(17,43),8)+random.sample(range(44,67),7)+random.sample(range(68,92),5)+random.sample(range(93,140),7)+random.sample(range(141,185),7)+random.sample(range(186,197),3)
    
    #4-6
    #idx_train=random.sample(range(1,16),6)+random.sample(range(17,43),10)+random.sample(range(44,67),9)+random.sample(range(68,92),9)+random.sample(range(93,140),18)+random.sample(range(141,185),18)+random.sample(range(186,197),4)    
    #idx_val=random.sample(range(1,16),3)+random.sample(range(17,43),5)+random.sample(range(44,67),5)+random.sample(range(68,92),5)+random.sample(range(93,140),6)+random.sample(range(141,185),6)+random.sample(range(186,197),2)
    
    #5-5
    #idx_train=random.sample(range(1,16),8)+random.sample(range(17,43),13)+random.sample(range(44,67),11)+random.sample(range(68,92),12)+random.sample(range(93,140),23)+random.sample(range(141,185),22)+random.sample(range(186,197),5)    
    #idx_val=random.sample(range(1,16),3)+random.sample(range(17,43),5)+random.sample(range(44,67),5)+random.sample(range(68,92),5)+random.sample(range(93,140),10)+random.sample(range(141,185),10)+random.sample(range(186,197),3)
    
    #6-4
    #idx_train=random.sample(range(1,16),10)+random.sample(range(17,43),15)+random.sample(range(44,67),15)+random.sample(range(68,92),15)+random.sample(range(93,140),30)+random.sample(range(141,185),30)+random.sample(range(186,197),7)    
    #idx_val=random.sample(range(1,16),5)+random.sample(range(17,43),5)+random.sample(range(44,67),5)+random.sample(range(68,92),5)+random.sample(range(93,140),10)+random.sample(range(141,185),10)+random.sample(range(186,197),3)
    
    #7-3
    #idx_train=random.sample(range(1,16),10)+random.sample(range(17,43),18)+random.sample(range(44,67),16)+random.sample(range(68,92),18)+random.sample(range(93,140),33)+random.sample(range(141,185),32)+random.sample(range(186,197),7)    
    #idx_val=random.sample(range(1,16),5)+random.sample(range(17,43),5)+random.sample(range(44,67),8)+random.sample(range(68,92),9)+random.sample(range(93,140),15)+random.sample(range(141,185),15)+random.sample(range(186,197),3)
    
    #8-2
    #idx_train=random.sample(range(1,16),12)+random.sample(range(17,43),21)+random.sample(range(44,67),18)+random.sample(range(68,92),19)+random.sample(range(93,140),38)+random.sample(range(141,185),35)+random.sample(range(186,197),9)    
    #idx_val=random.sample(range(1,16),6)+random.sample(range(17,43),10)+random.sample(range(44,67),9)+random.sample(range(68,92),8)+random.sample(range(93,140),20)+random.sample(range(141,185),20)+random.sample(range(186,197),6)
    
    #9-1
    idx_train=random.sample(range(1,16),14)+random.sample(range(17,43),23)+random.sample(range(44,67),21)+random.sample(range(68,92),22)+random.sample(range(93,140),42)+random.sample(range(141,185),39)+random.sample(range(186,197),10)    
    idx_val=random.sample(range(1,16),5)+random.sample(range(17,43),10)+random.sample(range(44,67),10)+random.sample(range(68,92),10)+random.sample(range(93,140),20)+random.sample(range(141,185),20)+random.sample(range(186,197),5)
    
    #print(len(idx_train))
    #print(len(idx_val))
    
    idx_train =  idx_train + random.sample(range(199,2906),2437)
    idx_val = idx_val + random.sample(range(199,2906),100)
    
    idx_test1 = range(1,2906)
    idx_test2 = list(set(idx_train).union(set(idx_val)))
    idx_test = list(set(idx_test1) ^ set(idx_test2))
    
    
    
    
    
    print(len(idx_train))
    print(len(idx_val))
    print(len(idx_test))
    
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test





def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    #print(output)
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
