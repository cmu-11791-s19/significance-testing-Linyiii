import numpy as np
from numpy import genfromtxt
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import csv


class SignificanceTesting(object):
    def __init__(self, filePath):
        self.ttestPValues = []
        self.wilcoxonPValues = []
        self.ksTestPValues = []
        self.meanDiffs = []
        self.means = []
        self.filePath = filePath
        self.loadData()

    def loadData(self):
        self.models_scores = ['Baseline_R2', 'Baseline+Fusion_R2', 'Baseline+Ordering_R2',
                              'Baseline+Ordering+Fusion_R2', 'Baseline_RSU4', 'Baseline+Fusion_RSU4',
                              'Baseline+Ordering_RSU4',
                              'Baseline+Ordering+Fusion_RSU4']
        self.data = genfromtxt(self.filePath, delimiter=',')[1:].T
        print self.data
        # print len(self.data)

    # the following three functions calculate p values for corresponding tests
    def ksTest(self, listA, listB):
        value, pvalue = ks_2samp(listA, listB)
        return pvalue

    def tTest(self, listA, listB):
        value, pvalue = ttest_ind(listA, listB)
        return pvalue

    def wilcoxonTest(self, listA, listB):
        T, pvalue = wilcoxon(listA, listB)
        return pvalue

    # the following three function get p-values for all pairs of systems for corresponding tests
    def doAllksTest(self):
        # ref = ['Baseline_R2' 0, 'Baseline+Fusion_R2' 1, 'Baseline+Ordering_R2' 2,
        #        'Baseline+Ordering+Fusion_R2' 3, 'Baseline_RSU4' 4, 'Baseline+Fusion_RSU4' 5,
        #        'Baseline+Ordering_RSU4' 6,
        #        'Baseline+Ordering+Fusion_RSU4' 7]
        # resultsData[1][1] = 'Baseline & Fusion' 0, 1
        # resultsData[2][1] = 'Baseline & Ordering' 0, 2
        # resultsData[3][1] = 'Fusion & Ordering' 1, 2
        # resultsData[4][1] = 'Baseline & Ordering+Fusion' 0, 3
        # resultsData[5][1] = 'Ordering & Ordering+Fusion' 2, 3
        # resultsData[6][1] = 'Fusion & Ordering+Fusion' 1, 3
        # resultsData[7][1] = 'Baseline & Fusion' 4, 5
        # resultsData[8][1] = 'Baseline & Ordering' 4, 6
        # resultsData[9][1] = 'Fusion & Ordering' 5, 6
        # resultsData[10][1] = 'Baseline & Ordering+Fusion' 4, 7
        # resultsData[11][1] = 'Ordering & Ordering+Fusion' 6, 7
        # resultsData[12][1] = 'Fusion & Ordering+Fusion' 5, 7
        pvalues = []
        pairs = [[0, 1], [0, 2], [1, 2], [0, 3], [2, 3], [1, 3], [4, 5], [4, 6],
                 [5, 6], [4, 7], [6, 7], [5, 7]]

        for i in range(12):
            pair = pairs[i]
            pvalue = self.ksTest(self.data[pair[0]], self.data[pair[1]])
            pvalues.append(pvalue)

        self.ksTestPValues = pvalues

    def doAlltTest(self):
        pvalues = []
        pairs = [[0, 1], [0, 2], [1, 2], [0, 3], [2, 3], [1, 3], [4, 5], [4, 6],
                 [5, 6], [4, 7], [6, 7], [5, 7]]

        for i in range(12):
            pair = pairs[i]
            pvalue = self.tTest(self.data[pair[0]], self.data[pair[1]])
            pvalues.append(pvalue)

        self.ttestPValues = pvalues

    def doAllwilcoxonTest(self):
        pvalues = []
        pairs = [[0, 1], [0, 2], [1, 2], [0, 3], [2, 3], [1, 3], [4, 5], [4, 6],
                 [5, 6], [4, 7], [6, 7], [5, 7]]

        for i in range(12):
            pair = pairs[i]
            pvalue = self.wilcoxonTest(self.data[pair[0]], self.data[pair[1]])
            pvalues.append(pvalue)

        self.wilcoxonPValues = pvalues

    def calcMeanDiffs(self):
        diffs = []
        pairs = [[0, 1], [0, 2], [1, 2], [0, 3], [2, 3], [1, 3], [4, 5], [4, 6],
                 [5, 6], [4, 7], [6, 7], [5, 7]]
        for i in range(12):
            pair = pairs[i]
            diff = self.means[pair[1]] - self.means[pair[0]]
            diffs.append(diff)

        self.meanDiffs = diffs

    def basicStats(self):
        # calculate mean, median, mode, min and max for all models
        means, medians, modes, mins, maxs = [], [], [], [], []
        for row in range(8):
            means.append(np.mean(self.data[row]))
        for row in range(8):
            medians.append(np.median(self.data[row]))
        for row in range(8):
            modes.append(stats.mode(self.data[row])[0])
        for row in range(8):
            mins.append(min(self.data[row]))
        for row in range(8):
            maxs.append(max(self.data[row]))

        self.means = means

        resultsFile = open('BasicStatsResults.csv', 'w')
        w = 7
        h = 9
        resultsData = [[0 for x in range(w)] for y in range(h)]
        resultsData[0] = ['metric', 'model', 'mean', 'median', 'mode', 'min', 'max']
        for row in range(1, 5):
            resultsData[row][0] = 'ROUGE-2'
        for row in range(5, 9):
            resultsData[row][0] = 'ROUGE-SU4'

        # assert (len(maxs) == 8)
        for row in range(8):
            resultsData[row + 1][2] = means[row]
            resultsData[row + 1][3] = medians[row]
            resultsData[row + 1][4] = modes[row]
            resultsData[row + 1][5] = mins[row]
            resultsData[row + 1][6] = maxs[row]

        resultsData[1][1] = 'Baseline'
        resultsData[2][1] = 'Baseline+Fusion'
        resultsData[3][1] = 'Baseline+Ordering'
        resultsData[4][1] = 'Baseline+Ordering+Fusion'
        resultsData[5][1] = 'Baseline'
        resultsData[6][1] = 'Baseline+Fusion'
        resultsData[7][1] = 'Baseline+Ordering'
        resultsData[8][1] = 'Baseline+Ordering+Fusion'

        with resultsFile:
            writer = csv.writer(resultsFile)
            writer.writerows(resultsData)
        resultsFile.close()

    def writeOutput(self):
        resultsFile = open('SigTestResults.csv', 'w')
        w = 6
        h = 13
        resultsData = [[0 for x in range(w)] for y in range(h)]
        resultsData[0] = ['metric', 'model', 'mean diff', 'P(T test)', 'P(wilcoxon test)', 'P(ks test)']
        for row in xrange(1, 7):
            resultsData[row][0] = 'ROUGE-2'
        for row in xrange(7, 13):
            resultsData[row][0] = 'ROUGE-SU4'

        # fill in p values
        for row in range(12):
            resultsData[row + 1][2] = self.meanDiffs[row]
            resultsData[row + 1][3] = self.ttestPValues[row]
            resultsData[row + 1][4] = self.wilcoxonPValues[row]
            resultsData[row + 1][5] = self.ksTestPValues[row]

        # add model names
        resultsData[1][1] = 'Baseline & Fusion'
        resultsData[2][1] = 'Baseline & Ordering'
        resultsData[3][1] = 'Fusion & Ordering'
        resultsData[4][1] = 'Baseline & Ordering+Fusion'
        resultsData[5][1] = 'Ordering & Ordering+Fusion'
        resultsData[6][1] = 'Fusion & Ordering+Fusion'
        resultsData[7][1] = 'Baseline & Fusion'
        resultsData[8][1] = 'Baseline & Ordering'
        resultsData[9][1] = 'Fusion & Ordering'
        resultsData[10][1] = 'Baseline & Ordering+Fusion'
        resultsData[11][1] = 'Ordering & Ordering+Fusion'
        resultsData[12][1] = 'Fusion & Ordering+Fusion'

        with resultsFile:
            writer = csv.writer(resultsFile)
            writer.writerows(resultsData)
        resultsFile.close()

    def boxingPlot(self):
        plt.figure()
        plt.boxplot(self.data.T)
        plt.show()


if __name__ == '__main__':
    filePath = "ROUGE_SCORES.csv"
    sigInstance = SignificanceTesting(filePath)
    sigInstance.basicStats()  # get mean, median, mode, min and max for each model
    sigInstance.calcMeanDiffs()
    sigInstance.doAllksTest()
    sigInstance.doAlltTest()
    sigInstance.doAllwilcoxonTest()
    sigInstance.writeOutput()
