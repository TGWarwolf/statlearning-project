目录结构
-kNN 存放kNN的预测数据和评价指标
-LogisticR 存放逻辑回归的预测数据和评价指标
-QDA 存放QDA的预测数据和评价指标
-SVM 存放SVM的预测数据和评价指标
-test 存放测试数据
-train 存放训练数据
label_train.csv 存放测试集的标签
main.py 主函数
kNN.py kNN的实现
LDA.py QDA的实现
LogisticR.py 逻辑回归的实现
SVM.py SVM的实现



参数说明：
--Alg           模型类型     可选：QDA, SVM, kNN, LR
--train_type                   可选：0 预测测试集的label， 1 在训练集进行模型评估
--PCA      PCA保留维数  int型
--K          kNN参数k       int型
--kernel  核函数类型      可选： rbf, linear, poly, sigmoid

调用实例：
使用线性核，在训练集上使用SVM进行评估：
python main.py --Alg=SVM --train_type=1  --kernel=linear
使用30-NN，PCA维数保留50，对测试集进行预测：
python main.py --Alg=kNN --train_type=0 --K=30  --PCA=50
