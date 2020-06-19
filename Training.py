import sys,os
import importlib
import torch
import MLP
import itertools
import numpy as np
importlib.reload(sys)
from torch.autograd import Variable

#################################################

# # # # # PATH # # # # #
full_path=os.path.realpath(__file__)
eop=full_path.rfind(__file__)
eop=full_path.rfind(os.path.basename(__file__))
main_path=full_path[0:eop]
folder_path=full_path[0:eop]+u'DATA'
mid_result_path=folder_path+u'/5.mid result/'


M_num=495
D_num=383

EPOCH=100
BATCH_SIZE=64

import os
meta_paths_filePath = 'DATA/6.meta path/'
fileList = os.listdir(meta_paths_filePath)
print(fileList)

import pandas as pd
train_x=[]
for file in fileList:
    meta_paths_pd = pd.read_csv(meta_paths_filePath + file, nrows=100, header=None)
    index_update=[]
    for ch in file[0:file.find('_')]:
        if ch == 'm':
            index_update.append(0)
        else:
            index_update.append(M_num)
    meta_paths_pd = meta_paths_pd + index_update
    meta_paths_list = meta_paths_pd.values.tolist()
    train_x = train_x + meta_paths_list



embed = np.loadtxt(mid_result_path + "embeding.txt", delimiter=',')
embed=torch.FloatTensor(embed)
embed = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embed))


m_d = np.loadtxt(mid_result_path + "m_d.txt", delimiter=',')
print(m_d.shape)


model_mlp = MLP.MLP()
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=1e-2)
loss_func_mlp = torch.nn.BCELoss()

for epoch in range(EPOCH):
    for i in range(0,len(train_x),BATCH_SIZE):
        var_x = train_x[i:i+BATCH_SIZE]
        lens = [len(x) for x in var_x]
        lens = torch.LongTensor(lens)

        batch = list(itertools.zip_longest(*var_x, fillvalue=0))
        batch = Variable(torch.LongTensor(batch))

        embed_batch = embed(batch)
        batch_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_batch, lens, enforce_sorted=False)



        com_embeding = []
        pairs_in_batch = []
        index_of_pair = []
        model_mlp = MLP.MLP()
        expected_out_mlp = []
        i=0
        for item in var_x:

            first_node = item[0]
            end_node = item[-1]
            expected_out_mlp.append(m_d[first_node][end_node-M_num])

            dot = embed.weight[first_node] * embed.weight[end_node]
            diff = embed.weight[first_node] - embed.weight[end_node]
            com_embeding.append(np.hstack((embed.weight[first_node], embed.weight[end_node], diff, dot)).tolist())#组合编码


        batch_mlp = Variable(torch.FloatTensor(com_embeding),requires_grad=True)
        out_mlp_embed_pair ,out_mlp = model_mlp(batch_mlp)

        optimizer_mlp.zero_grad()
        expected_out_mlp = Variable(torch.FloatTensor(expected_out_mlp))
        loss_mlp = loss_func_mlp(out_mlp, expected_out_mlp)
        loss_mlp.backward()
        optimizer_mlp.step()
        print("-----")
        print(loss_mlp)

