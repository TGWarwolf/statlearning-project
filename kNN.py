import numpy as np

class kNN:
    def __init__(self,class_number,K):
        self.class_number=class_number
        self.K=K
    def _Euclidean_dist(self,x,y):
        return np.sqrt(np.sum((x-y)**2))
    def predict_kNN(self,train_data,test_data,label_flag=0):
        result={}
        class_number=self.class_number
        [train_number,y]=np.shape(train_data[0])
        [sample_number,x]=np.shape(test_data[0])
        pre_label=np.zeros(sample_number)
        vote_list=np.zeros((sample_number,class_number))
        for i in range(sample_number):
            dist=np.zeros(train_number)
            for j in range(train_number):
                dist[j]=self._Euclidean_dist(train_data[0][j,:],test_data[0][i,:])
            #print(dist)
            idx = dist.argsort()[:self.K]
            #print(dist)
            
            for k in range(class_number):
                kidx=np.where(train_data[1][idx]==k)[0]
                vote_list[i,k]=np.size(kidx)
                #vote_list[i,k]=np.mean(1.0/(dist[kidx]+1))
            #print(vote_list[i,:])
            #input()
            pre_label[i]=np.argmax(vote_list[i,:])
        if(label_flag==0):
            for i in range(sample_number):
                result[test_data[1][i]]=np.argmax(vote_list[i,:])
        elif(label_flag==1):
            for i in range(sample_number):
                pre_label[i]=np.argmax(vote_list[i,:])
            F1=np.zeros(class_number)
            acc=np.zeros(class_number)
            for i in range(self.class_number):
                TP=np.size(pre_label[(pre_label[:]==i)&(test_data[1][:]==i)])
                FN=np.size(pre_label[(pre_label[:]!=i)&(test_data[1][:]==i)])
                FP=np.size(pre_label[(pre_label[:]==i)&(test_data[1][:]!=i)])
                TN=np.size(pre_label[(pre_label[:]!=i)&(test_data[1][:]!=i)])
                if((TP+FP==0)|(TP+FN==0)|(TP==0)):
                    continue
                precision=TP*1.0/(TP+FP)
                recall=TP*1.0/(TP+FN)
                F1[i]=precision*recall*2.0/(precision+recall)
                acc[i]=(TP+TN)*1.0/(TP+TN+FN+FP)
                
            result['acc']=np.mean(acc)
            result['F1']=np.mean(F1[F1!=0])
            
        return result