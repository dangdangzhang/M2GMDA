#
import sys,os
import numpy as np
from auto_encoder import AutoEncoder
import importlib
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable

importlib.reload(sys)
# sys.setdefaultencoding('gbk')
# sys.setdefaultencoding('utf-8')
#################################################

# # # # # PATH # # # # #
full_path=os.path.realpath(__file__)
eop=full_path.rfind(__file__)
eop=full_path.rfind(os.path.basename(__file__))
main_path=full_path[0:eop]
folder_path=full_path[0:eop]+u'DATA'
mid_result_path=folder_path+u'/5.mid result/'


def m_embeding():
    M = np.loadtxt(mid_result_path + "M.txt", delimiter=',')
    EPOCH = 10
    BATCH_SIZE = 64
    LR = 0.005
    autoencoder = AutoEncoder(M.shape[1])
    M = torch.tensor(M, dtype=torch.float32)
    M_train = Data.DataLoader(dataset=M, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    for epoch in range(EPOCH):
        for step, x in enumerate(M_train):
            # print(x)
            b_x = Variable(x)#
            b_y = Variable(x)
            encoded, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    encoded_data, _ = autoencoder(Variable(M))
    np.savetxt(mid_result_path + "m_coder.txt", encoded_data.detach().numpy(), delimiter=',', fmt='%.4f')
    print(encoded_data)



def d_embeding():
    D = np.loadtxt(mid_result_path + "D.txt", delimiter=',')
    EPOCH = 10
    BATCH_SIZE = 64
    LR = 0.005
    autoencoder = AutoEncoder(D.shape[1])
    D = torch.tensor(D, dtype=torch.float32)
    D_train = Data.DataLoader(dataset=D, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    for epoch in range(EPOCH):
        for step, x in enumerate(D_train):
            b_x = Variable(x)  #
            b_y = Variable(x)
            encoded, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    encoded_data, _ = autoencoder(Variable(D))
    np.savetxt(mid_result_path + "d_coder.txt", encoded_data.detach().numpy(), delimiter=',', fmt='%.4f')
    print(encoded_data)
