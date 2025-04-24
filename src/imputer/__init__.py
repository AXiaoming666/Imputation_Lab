from . import forward
from . import mean
from . import KNN
from . import XGBoost
from . import IIM

forward_impute = forward.forward_impute
mean_impute = mean.mean_impute
KNN_impute = KNN.KNN_impute
XGBoost_impute = XGBoost.XGBoost_impute
IIM_adaptive = IIM.IIM_adaptive

def impute(data, method="Mean", **kwargs):
    """
    统一的插补接口，根据方法名称自动调用对应的插补函数
    
    参数:
        data: 包含缺失值的数据集
        method: 插补方法名称，可选值: "Mean", "Forward", "KNN", "XGBoost", "IIM"
        **kwargs: 传递给具体插补方法的其他参数
    
    返回:
        插补后的数据集
    """
    imputers = {
        "Mean": mean_impute,
        "Forward": forward_impute,
        "KNN": KNN_impute,
        "XGBoost": XGBoost_impute,
        "IIM": IIM_adaptive
    }
    
    if method not in imputers:
        raise ValueError(f"不支持的插补方法: {method}。支持的方法有: {', '.join(imputers.keys())}")
    
    return imputers[method](data, **kwargs)