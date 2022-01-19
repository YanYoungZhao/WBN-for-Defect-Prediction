from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef, f1_score

from ImprovedSDA import ImprovedSDA
import LoadData
import Global
import numpy as np


if __name__ == '__main__':
    X1, X2 = LoadData.load_data(Global.SOURCE_FILE_NAME)
    Y1, Y2 = LoadData.load_data(Global.TARGET_FILE_NAME)
    Y = np.concatenate((Y1, Y2), axis=0)
    label = Y[:,-1]  # 获取标签
    isda = ImprovedSDA(X1, X2, Y, minSizeOfSubclass=35)
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
    # auc = my_AUC(label,predictions)
    F1 = f1_score(label, predictions),
    recall = recall_score(label, predictions)
    MCC = matthews_corrcoef(label, predictions)
    print('FPR=%f' % pf, 'recall=%f' % recall, 'F1=%f' %F1, 'MCC = %f' % MCC, 'AUC=%f' % auc)
    print(predictions)
    print(label)