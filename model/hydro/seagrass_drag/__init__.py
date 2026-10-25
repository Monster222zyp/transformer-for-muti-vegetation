"""柔性水草阻力计算模块，全部力的单位为 N。"""

# 从公共入口统一导出 API、数值设置、结果结构和可由调用者捕获的告警类型。
from .api import calculate_force, predict_drag
from .model import calculate_drag_coefficient_v2
from .types import ColumnDeformation, ConvergenceWarning, DragResult, NumericalOptions, NumericalStabilityWarning

__version__ = "0.1.0"
__all__ = ["calculate_force", "predict_drag", "calculate_drag_coefficient_v2",
           "NumericalOptions", "DragResult", "ColumnDeformation",
           "ConvergenceWarning", "NumericalStabilityWarning"]
