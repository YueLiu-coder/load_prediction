#需要导入的pyhon包
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import collections
import pandas as pd
import sklearn
from sklearn import tree
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# 用pandas读入csv文件
train = pd.read_csv('./loan_train.csv')
test = pd.read_csv('./loan_test.csv')

# 处理空值的数据
imp = Imputer(missing_values='NaN',strategy='most_frequent',verbose=0)
train = train.dropna()
test = test.dropna()

#去除id号
train = train.drop(['Loan_ID'],axis=1)
test = test.drop(['Loan_ID'],axis=1)

# 将类别型（categorical）属性映射成数值型属性
d = collections.defaultdict(LabelEncoder)
train = train.apply(lambda x: d[x.name].fit_transform(x))
test = test.apply(lambda x: d[x.name].fit_transform(x))

# 查看训练集和测试集中的部分数据和维度
# print("Train_data:")
# print(train.head())
# print("\nTrain_shape:")
# print(train.shape)
# print("\nTest_data:")
# print(test.head())
# print("\nTest_shape:")
# print(test.shape)

# 用pandas提取出训练集和测试集中的feature和target
train_x = train.drop('Loan_Status',axis=1)
train_y = train.Loan_Status
test_x = test.drop('Loan_Status',axis=1)
test_y = test.Loan_Status
# 查看训练集和测试集中feature和target的部分数据和维度

# print("\nTrain_x:")
# print(train_x.head())
# print("\nTrain_x_shape:")
# print(train_x.shape)

# print("\nTrain_y:")
# print(train_y.head())
# print("\nTrain_y_shape:")
# print(train_y.shape)

# print("\nTest_x:")
# print(test_x.head())
# print("\nTest_x_shape:")
# print(test_x.shape)

# print("\nTest_y:")
# print(test_y.head())
# print("\nTest_y_shape:")
# print(test_y.shape)
# print("\n")


# 定义两个算法的AUC列表来存储不同树最大层数的AUC
DTC_ENTROPY = []
DTC_GINI = []
# 决策树
# 生成决策树
for i in range(1,20):
    # 两种算法
    clf_c4_5 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=i)
    clf_CART = tree.DecisionTreeClassifier(criterion='gini', max_depth=i)
    #print("DecisionTreeClassifier:")
    #print(clf)
    # 喂入数据
    clf_c4_5 = clf_c4_5.fit(train_x, train_y)
    clf_CART = clf_CART.fit(train_x, train_y)
    # 测试
    test_pred1 = []
    test_pred2 = []
    for i in range(len(test_x)):
        # 取出test_x
        test_rec = test_x.iloc[i,:]
        # 预测的值
        test_pred1.append((clf_c4_5.predict([test_rec])[0]))
        test_pred2.append((clf_CART.predict([test_rec])[0]))

    #print("Prediction:")
    #print(np.array((test_pred)))
    #print("True:")
    #print(np.array(list(test_y)))
    # 计算预测结果的AUC(Area under curve):ROC曲线下的面积
    # ROC曲线：用不同的阀值，统计出一组不同阀值下的TPR（True positive rate）（真阳率）和FPR（True positive rate）（假阳率）的关系。
    auc1 = sklearn.metrics.roc_auc_score(np.array((test_pred1)),np.array(list(test_y)))
    auc2 = sklearn.metrics.roc_auc_score(np.array((test_pred2)), np.array(list(test_y)))
    #auc3 = sklearn.metrics.roc_auc_score(np.array((test_pred3)), np.array(list(test_y)))
    #print("AUC value:")
    #print(auc)
    #print(auc3)
    DTC_ENTROPY.append(auc1)
    DTC_GINI.append(auc2)


# 画图对比C4.5算法和CART算法
plt.figure(1)
plt.xlabel = 'MAX_DEPTH'
plt.ylabel = 'AUC'
plt.plot(range(1,20),DTC_ENTROPY)
plt.plot(range(1,20),DTC_GINI)
plt.show()

# 挑选出AUC最高的max_depth
#print(max(DTC_ENTROPY))
best_max_depth_1 = DTC_ENTROPY.index(max(DTC_ENTROPY))+1
#print(best_max_depth_1)

# 挑选出AUC最高的max_depth
#print(max(DTC_GINI))
best_max_depth_2 = DTC_GINI.index(max(DTC_GINI))+1
#print(best_max_depth_2)

print("BEST_MAX_DEPTH =",best_max_depth_1)
print("BEST_AUC1_VALUE:",max(DTC_ENTROPY))