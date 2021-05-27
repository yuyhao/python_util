"""
评分卡模型工具类
"""

import numpy  as np
import pandas as pd

class ScoreCard:
    def __init__(self, P, P0, PDO):
        self.P = P  # 违约的比率()odds
        self.P0 = P0  # 在违约概率为P下, 得分=P0
        self.PDO = PDO # 在违约概率为2P(翻倍), 得分的减少值
        self.B = self.PDO / np.log(2)
        self.A = self.P0 + self.B * np.log(self.P)

        """
        例
        诚信不良概率为5%时, 得分为1000
        诚信不良概率为10%时，得分为900(PDO = 100 = 1000 - 900)
        """

    """
    得分刻度表
    """
    def score_table(self):
        table_1 = []
        table_2 = []
        table_1_score = []
        table_2_score = []

        j = self.P
        s = self.P0
        for i in range(10):
            table_1.append(j)
            table_1_score.append(s)
            j = j * 2
            s -= self.PDO

        j2 = self.P/2
        s2 = self.P0 + self.PDO
        for j in range(5):
            table_2.append(j2)
            table_2_score.append(s2)
            j2 = j2 / 2
            s2 += self.PDO

        table_2.reverse()
        table_2_score.reverse()
        p = map(lambda x: round(x, 3), table_2 + table_1)
        score = table_2_score + table_1_score

        data = pd.DataFrame({'p':p, 'score':score})
        return data

    """
    通过概率计算得分
    """
    def get_score_by_prob(self, p):
        # odds
        odds = p / (1 - p)
        score = self.A - self.B * np.log(odds)

        return score

    """
    计算基础分
    """
    def get_base_score(self, intercept):
        """
        :param intercep: 逻辑回归截距项 float
        :return: float
        """
        return self.A - self.B * intercept


    """
    计算单个变量各分箱值的得分
    """
    def get_score_by_single_var(self, coea, woe):
        """
        :param coea: 变量的逻辑回归的系数 float
        :param woe: 变量各分箱的woe值 dict
        :return: 各分箱的分值 dict
        """

        score_dict = {}
        for key, value in woe.items():
            score = - 1 * coea * value * self.B
            score_dict[value] = score

        return score_dict

if __name__  == '__main__':
    score_card = ScoreCard(0.05, 1000, 100)
    score_card.A
    score_card.B








