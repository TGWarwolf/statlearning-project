import numpy as np
from sklearn import linear_model

class LogisticR:
    def __init__(self,class_number):
        self.class_number=class_number
        
    def _LR_train(self,train_data,i,j):
        if (i==j):
            return []
        else:
            idxi=np.where(train_data[1][:]==i)[0]
            idxj=np.where(train_data[1][:]==j)[0]
            
            isize=np.size(idxi)
            jsize=np.size(idxj)
            
            X=np.vstack((train_data[0][idxi,:],train_data[0][idxj,:]))
            y=np.vstack((np.zeros((isize,1)),np.ones((jsize,1))))
            y=y.flatten()
            LR_model=linear_model.LogisticRegression(C=1e5)
            LR_model.fit(X,y)
            return LR_model
    def one_one_LR_train(self,train_data):
        class_number=self.class_number
        LR_model=[]
        for i in range(class_number):
            LR_model.append([])
            for j in range(class_number):
                LR_model[i].append([])
                
        for i in range(class_number):
            for j in range(i+1,class_number):
                LR_model[i][j]=self._LR_train(train_data,i,j)
        return LR_model
    def one_one_LR_predict(self,train_data,test_data,label_flag=0):
        result={}
        class_number=self.class_number
        LR_model=self.one_one_LR_train(train_data)
        [sample_number,x]=np.shape(test_data[0])
        pre_label=np.zeros(sample_number)
        vote_list=np.zeros((sample_number,class_number))
        
        for i in range(class_number):
            for j in range(i+1,class_number):
                temp_result=LR_model[i][j].predict(test_data[0])
                vote_list[np.where(temp_result==0),i]+=1
                vote_list[np.where(temp_result==1),j]+=1
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