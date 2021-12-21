import numpy as np

class GaussianLDA:
    def __init__(self,class_number):
        #self.train_data=train_data
        #self.test_data=test_data
        self.class_number=class_number
        #[x,self.data_rank]=np.shape(train_data[0])
        '''
        self.mean=np.zeros((self.data_rank,self.class_number))
        self.var=np.zeros((self.data_rank,self.class_number))
        
        self.cov=np.zeros((self.data_rank,self.data_rank,class_number))
        self.cov_inv=np.zeros((self.data_rank,self.data_rank,class_number))
        
        self.prior=np.zeros(self.class_number)
        '''
        
        
    def _get_priors(self,train_data):
        class_number=self.class_number
        [x,data_rank]=np.shape(train_data[0])
        prior=np.zeros(class_number)
        for i in range(class_number):
            idx=np.where(train_data[1][:]==i)[0]
            prior[i]=np.size(train_data[1][idx])
        prior=prior/np.sum(prior)
        return prior
    
    def _get_means(self,train_data):
        class_number=self.class_number
        [x,data_rank]=np.shape(train_data[0])
        mean=np.zeros((data_rank,class_number))
        for i in range(class_number):
            idx=np.where(train_data[1][:]==i)[0]
            mean[:,i]=np.mean(train_data[0][idx,:])
        return mean
    '''
    def _get_vars(self):
        for i in range(self.class_number):
            idx=np.where(self.train_data[1][:]==i)[0]
            self.var[:,i]=np.var(self.train_data[0][idx,:])
        return self.var
    '''
    def _get_covs(self,train_data):
        class_number=self.class_number
        [x,data_rank]=np.shape(train_data[0])
        cov=np.zeros((data_rank,data_rank,class_number))
        cov_inv=np.zeros((data_rank,data_rank,class_number))
        for i in range(class_number):
            idx=np.where(train_data[1][:]==i)[0]
            cov[:,:,i]=np.cov(train_data[0][idx,:].T)
            cov_inv[:,:,i]=np.linalg.inv(cov[:,:,i])
        return cov,cov_inv
    
    def _get_likelihood_QDA(self,X,prior,mean,cov,cov_inv):
        class_number=self.class_number
        like=np.zeros(class_number)
        for i in range(class_number):
            delta=X-mean[:,i]
            #like[i]=np.log(self.prior[i])-np.sum(np.log(self.var[:,i]))-np.sum(0.5*(sample-self.mean[:,i])**2/self.var[:,i]**2)
            like[i]=np.log(prior[i])-0.5*np.log(np.linalg.det(cov[:,:,i]))-0.5*np.dot(delta,cov_inv[:,:,i]).dot(delta)
        
        return like
    '''
    def _get_likelihood_LDA(self,sample):
        like=np.zeros(self.class_number)
        for i in range(self.class_number):
            mu=self.mean[:,i]
            #like[i]=np.log(self.prior[i])-np.sum(np.log(self.var[:,i]))-np.sum(0.5*(sample-self.mean[:,i])**2/self.var[:,i]**2)
            like[i]=np.log(self.prior[i])+np.dot(sample,self.cov_inv[:,:,i]).dot(mu)-0.5*np.dot(mu,self.cov_inv[:,:,i]).dot(mu)
        
        return like
    '''
    def train(self,train_data):
        return self._get_priors(train_data),self._get_means(train_data),self._get_covs(train_data)
    
    def predict(self,train_data,test_data,label_flag=0):
        [prior,mean,[cov,cov_inv]]=self.train(train_data)
        [sample_number,x]=np.shape(test_data[0])
        result={}
        if (label_flag==0):
            
            pre_label=np.zeros(sample_number)
            for i in range(sample_number):
                #pre_label[i]=np.argmax(self._get_likelihood(self.test_data[0][i,:]))
                pre_label[i]=np.argmax(self._get_likelihood_QDA(test_data[0][i,:],prior,mean,cov,cov_inv))
                result[test_data[1][i]]=int(pre_label[i])
            
        elif(label_flag==1):
            F1=np.zeros(self.class_number)
            acc=np.zeros(self.class_number)
            pre_label=np.zeros(sample_number)
            for i in range(sample_number):
                #pre_label[i]=np.argmax(self._get_likelihood(self.test_data[0][i,:]))
                pre_label[i]=np.argmax(self._get_likelihood_QDA(test_data[0][i,:],prior,mean,cov,cov_inv))
                
            
            
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