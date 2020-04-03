# coding=utf-8
import sys,os
import numpy as np
import openpyxl as xlwt
import xlrd
import random
import datetime
import copy
import numba

# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve
reload(sys)
sys.setdefaultencoding('gbk')
# sys.setdefaultencoding('utf-8')
#################################################

# # # # # PATH # # # # #
full_path=os.path.realpath(__file__)
# print(os.path.basename(__file__))
eop=full_path.rfind(__file__)
eop=full_path.rfind(os.path.basename(__file__))
main_path=full_path[0:eop]
folder_path=full_path[0:eop]+u'DATA'
# # # # # GLOBALPARAMS # # # # #
# Constants
M_num=495                                           #miRNA num
D_num=383                                           #disease num
MD_num=5430                                         #associations num

# md
A=[['0'],['0']]                                     #dict for m-d
m_d=np.zeros(1)                                     #m-d associations mat
m_d_backup=np.zeros(1)                              #backup for m-d

# ss
ss=np.zeros(1)                                      #ss mat
ss_w=np.zeros(1)                                    #ss weighted mat，

# fs
fs=np.zeros(1)                                      #fs mat
fs_w=np.zeros(1)                                    #fs weighted mat

# predict result
y_local=np.zeros(1)                                 #global loocv results
y_global=np.zeros(1)                                #local loocv results

# index
Mindex=[]                                           #index for miRNA
Dindex=[]                                           #index for disease

######################################################################


# md
test_path_md=folder_path+u'/1.miRNA-disease associations/miRNA-disease_index.xlsx'

test_path_ss_1=folder_path+u'/2.disease semantic similarity 1/SS1.txt'
test_path_ss_2=folder_path+u'/3.disease semantic similarity 2/SS2.txt'
test_path_ss_w=folder_path+u'/2.disease semantic similarity 1/SSW1.txt'
# fs
test_path_fs=folder_path+u'/4.miRNA functional similarity/FS.txt'
test_path_fs_w=folder_path+u'/4.miRNA functional similarity/FSW.txt'
# Mindex
path_Mindex=folder_path+u'/1.miRNA-disease associations/miRNA_index.xlsx'
# Dindex
path_Dindex=folder_path+u'/1.miRNA-disease associations/disease_index.xlsx'
#临时输出路径
tmp_path=folder_path+u'/5.temp-result/'
######################################################################
# # # # # ALGORITHM START # # # # #

def __start__(path=folder_path,nm=M_num,nd=D_num):  
    global folder_path,M_num,D_num
    folder_path=path
    M_num=nm
    D_num=nd
    load_md(test_path_md)                           
    load_ss(test_path_ss_1,test_path_ss_2,test_path_ss_w)
    load_fs(test_path_fs,test_path_fs_w)
    load_index(path_Mindex,path_Dindex)


# # # # # DATA LOADER # # # # #

def load_md(path_md):
    global M_num, D_num, MD_num
    global A, m_d, m_d_backup

    # read xlsx
    md_table = xlrd.open_workbook(path_md)
    md_sheet = md_table.sheet_by_index(0)
    MD_num = md_sheet.nrows

    # save m-d associations in numpy
    m_d = np.zeros([M_num, D_num])  # init for m_d
    for i in range(MD_num):
        row_index = int(md_sheet.cell_value(rowx=i, colx=0)) - 1  
        col_index = int(md_sheet.cell_value(rowx=i, colx=1)) - 1
        A[0].append(int(row_index))
        A[1].append(int(col_index))
        m_d[row_index, col_index] = 1
    m_d_backup = copy.copy(m_d)
    del A[0][0], A[1][0] 


def load_ss(path_ss_1, path_ss_2, path_ss_w):
    global ss, ss_w

    # read ss into numpy array
    ss_1 = np.loadtxt(path_ss_1)
    ss_2 = np.loadtxt(path_ss_2)
    ss_w = np.loadtxt(path_ss_w)

    ss = (ss_1 + ss_2) / 2


def load_fs(path_fs, path_fs_w):
    global fs, fs_w

    # read fs into numpy array
    fs = np.loadtxt(path_fs)
    fs_w = np.loadtxt(path_fs_w)


def load_index(path_Mindex, path_Dindex):
    global M_num, D_num
    global Mindex, Dindex

    Mindex_table = xlrd.open_workbook(path_Mindex)
    Dindex_table = xlrd.open_workbook(path_Dindex)
    Mindex_sheet = Mindex_table.sheet_by_index(0)
    Dindex_sheet = Dindex_table.sheet_by_index(0)

    # read Mindex
    for i in range(M_num):
        data = str(Mindex_sheet.cell_value(rowx=i, colx=1))
        Mindex.append(data)

    # read Dindex
    for i in range(D_num):
        data = str(Dindex_sheet.cell_value(rowx=i, colx=1))
        Dindex.append(data)

######################################################################
# # # NUMBA ACCELERATE # # #

@numba.jit
def jitsum(x):                                      #sum by row
    [m,n]=x.shape
    s=np.zeros(m)
    for i in range(int(m)):
        for j in range(int(n)):
            s[i]+=x[i,j]
    return s
@numba.jit
def jitsumt(x):                                     #sum by col
    [m,n]=x.shape
    s=np.zeros(n)
    for i in range(int(m)):
        for j in range(int(n)):
            s[j]+=x[i,j]
    return s
@numba.jit
def jitsumall(x):                                   #sum all entries
    [m,n]=x.shape
    s=0
    for i in range(int(m)):
        for j in range(int(n)):
            s+=x[i,j]
    return s

# # # # # PREPROCESS # # # # #
@numba.jit
def __init__():  

    global M_num, D_num
    global ss, ss_w, fs, fs_w, m_d

    ''' Guassian Profile similarity into gs_m and gs_d '''
    # Gamma
    Gamma_d_s = 1
    Gamma_m_s = 1
    md_f = jitsumall(m_d * m_d)  
    Gamma_d = Gamma_d_s / (md_f / D_num)
    Gamma_m = Gamma_m_s / (md_f / M_num) 

  
    IP_d = np.tile(m_d.T, [1, D_num]) - np.resize(m_d.T, [1, M_num * D_num]) 
    IP_d = np.resize(IP_d * IP_d, [D_num * D_num, M_num]) 
    gs_d = np.exp(-Gamma_d * np.resize(jitsum(IP_d), [D_num, D_num]))  
    。

   
    IP_m = np.tile(m_d, [1, M_num]) - np.resize(m_d, [1, M_num * D_num])  
    IP_m = np.resize(IP_m * IP_m, [M_num * M_num, D_num]) 
    gs_m = np.exp(-Gamma_m * np.resize(jitsum(IP_m), [M_num, M_num]))

    ''' Integration ''' 
    m_m = fs * fs_w + (1 - fs_w) * gs_m  # SS for 1 GS for 0
    d_d = ss * ss_w + (1 - ss_w) * gs_d  # FS for 1 GS for 0

    # cal Jaccard similarity
    m_j = np.ones([M_num, M_num])
    d_j = np.ones([D_num, D_num])
    # retrive i and j-th row in m_d
    Mi = np.dot(m_d, m_d.T)  # intersect
    Di = np.dot(m_d.T, m_d)
    Mu_t = np.tile(m_d, [1, M_num]) + np.resize(m_d, [1, M_num * D_num])  # union
    Mu_t = np.resize(Mu_t, [M_num * M_num, D_num])
    Mu = np.resize(jitsum(Mu_t), [M_num, M_num]) - Mi
    Du_t = np.tile(m_d.T, [1, D_num]) + np.resize(m_d.T, [1, M_num * D_num])  
    Du_t = np.resize(Du_t, [D_num * D_num, M_num])
    Du = np.resize(jitsum(Du_t), [D_num, D_num]) - Di
    m_j = Mi / (Mu + (Mu == 0))
    d_j = Di / (Du + (Du == 0))  

    # Enhansed repression
    M = np.column_stack((m_m, m_j))  
    D = np.column_stack((d_d, d_j))

    return M, D,m_m,d_d
########################################################################

def indirect_3L(mat):
    m_d_3L=np.dot(np.dot(m_d,m_d.T),m_d)
    np.savetxt(tmp_path+"md3l.txt",m_d_3L,delimiter=',',fmt='%.5f')
    return m_d_3L

def indirect_mmd(m_m,m_d):
    mmd=np.dot(m_m,m_d)
    np.savetxt(tmp_path + "mmd.txt", mmd, delimiter=',', fmt='%.5f')
    return mmd

def indirect_mdm(m_d,d_d):
    mdd = np.dot(m_d, d_d)
    np.savetxt(tmp_path + "mdd.txt", mdd, delimiter=',', fmt='%.5f')
    return mdd

def m_d_degree(m_d,D_num,M_num):
    m_d_m_degree=np.tile(jitsumt(m_d.T),[D_num,1]).T
    m_d_d_degree = np.tile(jitsumt(m_d), [M_num, 1])
    m_d_degree_sum=m_d_m_degree+m_d_d_degree
    m_d_degree_mult = np.sqrt(m_d_m_degree * m_d_d_degree)
    return m_d_degree_sum,m_d_degree_mult


########################################################################
# # # # # # Output # # # # # #

def predict():
    global M_num, D_num, MD_num
    global m_d
    scores_unknown = []

    m_d = copy.copy(m_d_backup)
    for i in range(m_d.shape[1]):
        m_d[i][326]=0
    [M, D,m_m,d_d] = __init__()
    mmdegree = m_m_degree(m_m, D_num)
    dddegree = d_d_degree(d_d, M_num)
    m_d_3L=indirect_3L(m_d)
    mmd=indirect_mmd(m_m,m_d)/mmdegree
    mdm=indirect_mdm(m_d,d_d)/dddegree
    mmd=(mmd -np.min(mmd))  / (np.max(mmd)-np.min(mmd))
    mdm = (mdm - np.min(mdm)) / (np.max(mdm) - np.min(mdm))
    m_d_degree_sum,m_d_degree_mult = m_d_degree(m_d,D_num,M_num)
    # a=m_d_3L / m_d_degree_sum
    a = m_d_3L * m_d_degree_mult
    yc = np.dot(m_m,a)+np.dot(a,d_d)
    # m_d_final=np.zeros([M_num,D_num])+(yc!=0)

    # m_d_final = yc / np.max(yc) 
    m_d_final=(yc -np.min(yc))  / (np.max(yc)-np.min(yc))
    m_d_final=(m_d_final+mmd+mdm)/3

    m_d_origin=copy.copy(m_d_backup)
    np.savetxt(tmp_path + "m_d1.txt", m_d, delimiter=',', fmt='%d')
    np.savetxt(tmp_path + "m_d_our.txt", m_d_final, delimiter=',', fmt='%.5f')

    # fpr, tpr, theshold = roc_curve(m_d_origin.resize(1,M_num*D_num),m_d_final.resize(1,M_num*D_num))
    # plt.plot(fpr, tpr)
    count = 0
    for j in range(M_num):
        for k in range(D_num):
            if m_d[j, k] == 0:
                count += 1
                M_name = Mindex[j]
                D_name = Dindex[k]
                scores_unknown.append((D_name, M_name, yc[j, k]))
    scores_unknown = sorted(scores_unknown, key=lambda scores_unknown: scores_unknown[2], reverse=True)

    scores_table = xlwt.Workbook()
    scores_sheet = scores_table.active
    for i in range(count):
        scores_sheet.append(scores_unknown[i])
    save_path = main_path + u'/RESULT/Result_3L_only.xlsx'
    scores_table.save(save_path)
# # # # # TEST # # # # #
def predict2(m_d,m_m,d_d):
    global M_num, D_num, MD_num
    mmdegree = m_m_degree(m_m, D_num)
    dddegree = d_d_degree(d_d, M_num)
    m_d_3L = indirect_3L(m_d)
    mmd = indirect_mmd(m_m, m_d) / mmdegree
    mdm = indirect_mdm(m_d, d_d) / dddegree
    mmd = (mmd - np.min(mmd)) / (np.max(mmd) - np.min(mmd))  
    mdm = (mdm - np.min(mdm)) / (np.max(mdm) - np.min(mdm))
    m_d_degree_sum, m_d_degree_mult = m_d_degree(m_d, D_num, M_num)
    # a=m_d_3L / m_d_degree_sum
    a = m_d_3L * m_d_degree_mult
    yc = np.dot(m_m, a) + np.dot(a, d_d)  
    # m_d_final=np.zeros([M_num,D_num])+(yc!=0)

    # m_d_final = yc / np.max(yc)  
    m_d_final = (yc - np.min(yc)) / (np.max(yc) - np.min(yc))  
    m_d_final = (m_d_final + mmd + mdm) / 3

    m_d_origin = copy.copy(m_d_backup) 
    return m_d_final

# def global_loocv():
#     m_d = copy.copy(m_d_backup)
#     [M, D, m_m, d_d] = __init__()
#     m_d_final = predict2(m_d, m_m, d_d)
#     m_d_test=np.zeros([M_num, D_num])
#     index = np.argwhere(m_d == 1)
#     for i in index :
#         m_d[x,y]=0
#         m_d_final[x,y]=0
#         m_d_global=predict2(m_d,m_m,d_d)
#         m_d_test[x,y]=m_d_global[x,y]
#         m_d = copy.copy(m_d_backup)
#     m_d_final=m_d_final+m_d_test


''' RUN '''
def __run__(k):
    __start__()
    print 'data loaded\n'
    predict(k)
    m_d = copy.copy(m_d_backup)
    [M, D, m_m, d_d] = __init__()
    m_d_final_global=predict2(m_d,m_m,d_d)
    np.savetxt(tmp_path + "m_d_our_global.txt", m_d_final_global, delimiter=',', fmt='%.5f')
    print 'done\n'

''' MAIN '''
if __name__ == '__main__':
    if len(sys.argv) == 1:
        k=100
    else:
        k=sys.argv[1]
    __run__(k)
