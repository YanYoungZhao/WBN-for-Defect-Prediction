import le as le
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef, f1_score

from ImprovedSDA import ImprovedSDA
import LoadData
import Global
import numpy as np

'''
def my_AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]
    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i]== pre[j]:
                auc += 0.5
    return auc / (len(pos) * len(neg))
'''


if __name__ == '__main__':
    X1, X2 = LoadData.load_data(Global.FILE_NAME) # X1 defective instances & X2 defect-free instances
    n1 = len(X1)
    n2 = len(X2)
 #   number_defect = round(n1/2)   #分割的训练集
  #  number_free   = round(n2/2)  # 分割的训练集
 #   Y1, X1 = X1[n1-number_defect: n1], X1[: n1-number_defect] # Y1 是从 X1 （缺陷类中）中分割出来的测试集
 #   Y2, X2 = X2[n2-number_free: n2], X2[: n2-number_free] # Y1 是从 X1 （无缺陷类中）中分割出来的测试集 X1 X2是训练集
    number=5 # 分割的训练集
    Y1, X1 = X1[n1 - number: n1], X1[: n1 - number]  # Y1 是从 X1 （缺陷类中）中分割出来的测试集
    Y2, X2 = X2[n2 - number: n2], X2[: n2 - number]  # Y1 是从 X1 （无缺陷类中）中分割出来的测试集 X1 X2是训练集

    Y = np.concatenate((Y1, Y2), axis=0)  # 对数据进行拼接,训练集，axis： &0对行进行拼接 &1 是对列进行拼接
    label = Y[:,-1] #获取标签
    isda = ImprovedSDA(X1, X2, Y, minSizeOfSubclass=8)
    predictions = isda.within_predict()
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i, j in enumerate(label):
        if j:
            if predictions[i]:
                TP += 1
            else:
                FN += 1
        else:
            if predictions[i]:
                FP += 1
            else:
                TN += 1
    if (FP + TN) == 0:
        pf = "no negative samples."
    else:
        pf = FP / (FP + TN)
    try:
        auc = roc_auc_score(label, predictions)
    except ValueError as e:
        auc = str(e)
    #auc = my_AUC(label,predictions)
    F1= f1_score(label,predictions),
    recall = recall_score(label,predictions)
    MCC = matthews_corrcoef(label,predictions)
    print('FPR=%f' % pf,'recall=%f' %recall,'F1=%f' %F1, 'MCC = %f' % MCC, 'AUC=%f' %auc)
    print(predictions)
    print(label)

# done!




