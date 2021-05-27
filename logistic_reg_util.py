"""
逻辑回归
"""

from sklearn.linear_model import LogisticRegression
from util.metrics_util import MetricsUtil
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek
import pandas as pd

class LogisticRegUtil:

    """
    逻辑回归-不分训练集和测试集
    """
    @staticmethod
    def logit_reg_model_sample(X, y):
        logit_model = LogisticRegression()

        # 模型训练
        logit_model.fit(X, y)

        # 预测
        y_pre = logit_model.predict(X)

        # 评价指标
        metric_val = MetricsUtil.metrics_class(y, y_pre)

        # 参数
        coea = {}
        coea['intercept'] = logit_model.intercept_
        coea['param'] = logit_model.coef_

        return logit_model, coea, metric_val

    """
    逻辑回归-训练集和测试集
    """
    @staticmethod
    def logit_reg_model(X_train, X_test, y_train, y_test):
        logit_model = LogisticRegression()

        # 模型训练
        logit_model.fit(X_train, y_train)

        # 预测
        y_pre = logit_model.predict(X_test)
        y_pre = pd.Series(y_pre)

        # 评价指标
        metric_val = MetricsUtil.metrics_class(y_test, y_pre)

        # 参数
        coea = {}
        coea['intercept'] = logit_model.intercept_
        coea['param'] = logit_model.coef_

        return logit_model, coea, metric_val

    """
    逻辑回归-网格搜索法
    """
    @staticmethod
    def logit_reg_model_grid(X, y):
        # 网格参数设置
        param = {
            'C':[0.01, 0.1, 0.5, 1, 2, 10],
            'max_iter':[50, 100, 150, 200, 200]
        }

        # 基础模型
        logit_model = LogisticRegression()

        # 模型训练
        gs_model = GridSearchCV(logit_model, param, cv = 10)
        gs_model.fit(X, y)

        # 最佳模型
        best_logit_model = gs_model.best_estimator_

        # 预测
        y_pre = best_logit_model.predict(X)

        # 评价指标
        metric_val = MetricsUtil.metrics_class(y, y_pre)

        # 参数
        coea = {}
        coea['intercept'] = best_logit_model.intercept_
        coea['param'] = best_logit_model.coef_

        return best_logit_model, coea, metric_val

    """
    不均衡采样
    """
    @staticmethod
    def SMOTETomek_sample(X,y):
        smote_tomek = SMOTETomek(random_state=1234)
        x_s, y_s = smote_tomek.fit_resample(X, y)

        return x_s, y_s

if __name__ == '__main__':
    pass




