"""
评价指标
"""
from sklearn import metrics

class MetricsUtil:

    """
    分类评价指标
    """
    @staticmethod
    def metrics_class(y_true, y_pre):
        metric_dict = {}
        metric_dict['acc'] = metrics.accuracy_score(y_true, y_pre)
        metric_dict['recall'] = metrics.recall_score(y_true, y_pre)
        metric_dict['precision'] = metrics.precision_score(y_true, y_pre)
        metric_dict['auc_score'] = metrics.roc_auc_score(y_true, y_pre)
        metric_dict['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pre)
        return metric_dict