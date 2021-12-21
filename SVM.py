import numpy as np
from sklearn.svm import SVC

class SVM:
    def __init__(self,train_data,test_data,class_number,kernel='linear'):
        self.train_data=train_data
        self.test_data=test_data
        self.class_number=class_number
        [x,self.data_rank]=np.shape(train_data[0])
        [self.test_number,y]=np.shape(test_data[0])
        self.SVM_model=[]
        
        #print(np.shape(self.train_data[0]))
        
        self.kernel = kernel
    
    def _SVM_train(self,train_data,i,j):
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
            SVM_model=SVC(kernel=self.kernel)
            SVM_model.fit(X,y)
            return SVM_model
    '''
    def _SVM_train_m(self,i):
        idxi=np.where(self.train_data[1][:]==i)[0]
        idxj=np.where(self.train_data[1][:]!=i)[0]
        
        isize=np.size(idxi)
        jsize=np.size(idxj)
        
        X=np.vstack((self.train_data[0][idxi,:],self.train_data[0][idxj,:]))
        y=np.vstack((np.ones((isize,1)),np.zeros((jsize,1))))
        y=y.flatten()
        temp=SVC(kernel=self.kernel)
        temp.fit(X,y)
        return temp
    '''
    
    def one_one_SVM_train(self,train_data):
        class_number=self.class_number
        SVM_model=[]
        for i in range(class_number):
            SVM_model.append([])
            for j in range(class_number):
                SVM_model[i].append([])
                
        for i in range(class_number):
            for j in range(i+1,class_number):
                SVM_model[i][j]=self._SVM_train(train_data,i,j)
        return SVM_model
    '''
    def one_m_SVM_train(self):
        for i in range(self.class_number):
            self.SVM_model.append([])
                
        for i in range(self.class_number):
            self.SVM_model[i]=self._SVM_train_m(i)
    '''
                
    def one_one_SVM_predict(self,train_data,test_data,label_flag=0):
        result={}
        class_number=self.class_number
        SVM_model=self.one_one_SVM_train(train_data)
        [sample_number,x]=np.shape(test_data[0])
        pre_label=np.zeros(sample_number)
        vote_list=np.zeros((sample_number,class_number))
        
        for i in range(class_number):
            for j in range(i+1,class_number):
                temp_result=SVM_model[i][j].predict(test_data[0])
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
        
    '''
    def one_m_SVM_predict(self):
        result={}
        vote_list=np.zeros((self.test_number,self.class_number))
        for i in range(self.class_number):
            temp_result=self.SVM_model[i].predict(self.test_data[0])
            vote_list[np.where(temp_result==1),i]+=1
        for i in range(self.test_number):
            result[self.test_data[1][i]]=np.argmax(vote_list[i,:])
        return result
    '''