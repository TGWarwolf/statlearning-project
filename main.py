import csv,os
import numpy as np
import LDA
import SVM
import kNN
import LogisticR as LR
import argparse

class_number=20
raw_size=1500


def PCA(data, max_size, raw_size=1500):
    data = np.array(data)

    # 均值
    mean_vector = np.mean(data, axis=0)

    # 协方差
    cov_mat = np.cov(data - mean_vector, rowvar=0)

    # 特征值 特征向量
    fvalue, fvector = np.linalg.eig(cov_mat)
    [U,S,V]=np.linalg.svd(cov_mat)

    remain=np.zeros(max_size)
    # 排序
    fvaluesort = np.argsort(-fvalue)
    
    for i in range(max_size):
        remain[i]=np.sum(S[fvaluesort[:i]])/np.sum(S[0:raw_size])
    n=np.argmax(remain)+1
    
    print(n,remain[n-1])
    # 取前几大的序号
    fValueTopN = fvaluesort[:n]

    # 保留前几大的数值
    newdata = fvector[:, fValueTopN]

    #new = np.dot(data, newdata)

    return newdata,n



def open_label(file_path,file_name):
    '''
    load labels as dict
    '''
    with open(file_path+file_name,'r',encoding="utf-8") as f:
        dist={}
        reader = csv.reader(f)
        fieldnames = next(reader)
        csv_reader = csv.DictReader(f,fieldnames=fieldnames)
        label={}
        for row in csv_reader:
            d={}
            for k,v in row.items():
                d[k]=v
            label[d[fieldnames[0]]]=d[fieldnames[1]]
        return label
def open_data(data_path,data_number,data_size,class_number,label=[]):
    '''
    load data
    test data: data 1334*1500
    train data: data[0] 1334*1500
    '''
    i=0
    if class_number==0:
        data=[np.zeros((data_number,data_size)),[]]
        for path,dir_list,file_list in os.walk(data_path):
            for file_name in file_list:
                data[0][i,:]=np.load(data_path+file_name).flatten()
                data[1].append(file_name)
                i+=1
    else:
        data=[np.zeros((data_number,data_size)),np.zeros(data_number)]
        for path,dir_list,file_list in os.walk(data_path):
            for file_name in file_list:
                data[0][i,:]=np.load(data_path+file_name).flatten()
                data[1][i]=int(label[file_name])
                i+=1
                #data[int(label[file_name])].append(np.load(data_path+file_name).flatten())
    return data

def QDA_predict(PCA_max_size,train_data,test_data,predict_type=0,train_rate=0.8):

    [map_matrix,PCA_size]=PCA(train_data[0],PCA_max_size)

    train_data[0]=np.real(np.dot(train_data[0],map_matrix))
    
    test_data[0]=np.real(np.dot(test_data[0],map_matrix))
    
    #model_0.train(train_data,test_data,predict_type)
    model_0=LDA.GaussianLDA(class_number)
    if(predict_type==0):
        result=model_0.predict(train_data,test_data,predict_type)
        write_result("./QDA/","predict_QDA_PCA"+str(PCA_size)+".csv",result)
    elif(predict_type==1):
        [total_number,x]=np.shape(train_data[0])
        train_number=int(np.floor(total_number*train_rate))
        idx=np.random.choice(total_number, train_number, replace=False)
        idx_com=np.array(list(set(range(train_number))-set(idx)))
        result=model_0.predict([train_data[0][idx,:],train_data[1][idx]],[train_data[0][idx_com,:],train_data[1][idx_com]],predict_type)
        write_result("./QDA/","acc_QDA_PCA"+str(PCA_size)+".csv",result)

def SVM_predict(kernel,train_data,test_data,predict_type=0,train_rate=0.8):
    '''
    [map_matrix,PCA_size]=PCA(train_data[0],PCA_max_size)

    train_data[0]=np.real(np.dot(train_data[0],map_matrix))
    
    test_data[0]=np.real(np.dot(test_data[0],map_matrix))
    '''
    if(predict_type==0):
        SVM_model=SVM.SVM(train_data,test_data,class_number,kernel=kernel)
        result=SVM_model.one_one_SVM_predict(train_data,test_data,predict_type)
        write_result("./SVM/","predict_SVM_"+kernel+".csv",result)
    elif(predict_type==1):
        [total_number,x]=np.shape(train_data[0])
        train_number=int(np.floor(total_number*train_rate))
        idx=np.random.choice(total_number, train_number, replace=False)
        idx_com=np.array(list(set(range(train_number))-set(idx)))
        SVM_model=SVM.SVM(train_data,test_data,class_number,kernel=kernel)
        result=SVM_model.one_one_SVM_predict([train_data[0][idx,:],train_data[1][idx]],[train_data[0][idx_com,:],train_data[1][idx_com]],predict_type)
        write_result("./SVM/","acc_SVM_"+kernel+".csv",result)

def kNN_predict(PCA_max_size,train_data,test_data,K,predict_type=0,train_rate=0.8):
    [map_matrix,PCA_size]=PCA(train_data[0],PCA_max_size)

    train_data[0]=np.real(np.dot(train_data[0],map_matrix))
    
    test_data[0]=np.real(np.dot(test_data[0],map_matrix))
    model_2=kNN.kNN(class_number,K)
    if(predict_type==0):
        
        result=model_2.predict_kNN(train_data,test_data,predict_type)
        write_result("./KNN/","predict_"+str(K)+"kNN_PCA"+str(PCA_size)+".csv",result)
    elif(predict_type==1):
        [total_number,x]=np.shape(train_data[0])
        train_number=int(np.floor(total_number*train_rate))
        idx=np.random.choice(total_number, train_number, replace=False)
        idx_com=np.array(list(set(range(train_number))-set(idx)))
        result=model_2.predict_kNN([train_data[0][idx,:],train_data[1][idx]],[train_data[0][idx_com,:],train_data[1][idx_com]],predict_type)
        write_result("./KNN/","acc_"+str(K)+"kNN_PCA"+str(PCA_size)+".csv",result)

def LR_predict(PCA_max_size,train_data,test_data,predict_type=0,train_rate=0.8):
    [map_matrix,PCA_size]=PCA(train_data[0],PCA_max_size)

    train_data[0]=np.real(np.dot(train_data[0],map_matrix))
    
    test_data[0]=np.real(np.dot(test_data[0],map_matrix))
    
    if(predict_type==0):
        LR_model=LR.LogisticR(class_number)
        result=LR_model.one_one_LR_predict(train_data,test_data,predict_type)
        write_result("./LogisticR/","predict_LR_PCA"+str(PCA_size)+".csv",result)
    elif(predict_type==1):
        [total_number,x]=np.shape(train_data[0])
        train_number=int(np.floor(total_number*train_rate))
        idx=np.random.choice(total_number, train_number, replace=False)
        idx_com=np.array(list(set(range(train_number))-set(idx)))
        LR_model=LR.LogisticR(class_number)
        result=LR_model.one_one_LR_predict([train_data[0][idx,:],train_data[1][idx]],[train_data[0][idx_com,:],train_data[1][idx_com]],predict_type)
        write_result("./LogisticR/","acc_LR_PCA"+str(PCA_size)+".csv",result)


def write_result(file_path,file_name,result):
    with open(file_path+file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','category'])
        for k, v in result.items():
           writer.writerow([k, v])
    

data_path="./train/"
test_path="./test/"
label_path="./"

label=open_label(label_path,"label_train.csv")
train_data=open_data(data_path,1334,raw_size,class_number,label)
test_data=open_data(test_path,348,raw_size,0,[])
parser = argparse.ArgumentParser()
parser.add_argument('--Alg', type=str,default='QDA')
parser.add_argument('--train_type', type=int,default=0)
parser.add_argument('--PCA', type=int,default=30)
parser.add_argument('--K', type=int,default=30)
parser.add_argument('--kernel', type=str,default='rbf')
args = parser.parse_args()


PCA_size=args.PCA
K=args.K
kernel=args.kernel
train_type=args.train_type

if(args.Alg=='QDA'):
    QDA_predict(PCA_size,train_data,test_data,train_type)
elif(args.Alg=='SVM'):
    SVM_predict(kernel,train_data,test_data,train_type)
elif(args.Alg=='kNN'):
    kNN_predict(PCA_size,train_data,test_data,K,train_type)
elif(args.Alg=='LR'):
    LR_predict(PCA_size,train_data,test_data,train_type)
else:
    print('Please choose a correct model!')


