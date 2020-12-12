# -*- coding:utf8 -*-
# @TIME     : 2020/12/10 11:00
# @Author   : SuHao
# @File     : evaluate.py

'''
评估指标
'''

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Evaluate():
    def __init__(self, path):
        '''
        ROC: fpr-tpr
        PRC: precisoin-recall
        labels: true labels
        score: predicted probability
        '''
        self.scores = []
        self.labels = []
        self.path = path
        self.fpr = []
        self.tpr = []
        self.precision = []
        self.recall = []
        self.auc = 0.0
        self.best_predictions = []
        self.best_accuracy = 0.0
        self.best_thre = 0.0
        self.best_F1_score = 0.0             # F1_score corresponding with best_thre


    def cal_ROC(self):
        self.fpr, self.tpr, _ = roc_curve(self.labels, self.scores)

    def cal_PRC(self):
        self.precision, self.recall, _ = precision_recall_curve(self.labels, self.scores)

    def cal_auc(self):
        self.auc = roc_auc_score(self.labels, self.scores)

    def cal_best_threshold(self):
        precs, recs, thrs = precision_recall_curve(self.labels, self.scores)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        self.best_thre = thrs[np.argmax(f1s)]
        self.best_F1_score = np.max(f1s)
        self.best_predictions = [1 if i > self.best_thre else 0 for i in self.scores]
        self.best_accuracy = accuracy_score(self.labels, self.best_predictions)

    def plot(self):
        plt.figure(300)
        plt.plot(self.fpr, self.tpr)
        plt.title("ROC curve")
        plt.savefig(self.path+"/ROC.png")
        plt.close()

        plt.figure(300)
        plt.plot(self.precision, self.recall)
        plt.title("PRC curve")
        plt.savefig(self.path+"/PRC.png")
        plt.close()

    def run(self):
        self.cal_ROC()
        self.cal_PRC()
        self.cal_auc()
        self.cal_best_threshold()
        self.plot()
        print("auc: ", self.auc)
        print("best_accuracy: ", self.best_accuracy)
        print("best_thre: ", self.best_thre)
        print("best_F1_score: ", self.best_F1_score)


def draw_heatmap(residule):
    residule = (residule - np.min(residule)) / (np.max(residule) - np.min(residule)) * 255
    residule = residule.astype("uint8")
    residule = np.transpose(residule[0], [1, 2, 0])
    if residule.shape[2] == 3:
        residule = cv2.cvtColor(residule, cv2.COLOR_BGR2GRAY)
    elif residule.shape[2] == 1:
        residule = residule[:, :, 0]
    residule = cv2.applyColorMap(residule, cv2.COLORMAP_OCEAN)
    return residule
