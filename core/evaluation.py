import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class Evaluator:
    def __init__(self, log_callback):  # 增加日志回调参数
        self.log = log_callback  # 绑定到界面输出
        self.time_records = {}

    def _log(self, message, indent=0):
        """统一日志输出方法"""
        prefix = "|  " * indent
        self.log(f"{prefix}{message}")  # 调用界面回调

    def compute_dependency(self, data, attributes, decision_col):
        self._log("\n=== 依赖度计算 ===")
        # 检查属性集合有效性
        if not attributes:
            raise ValueError("属性集合不能为空")

        # 检查属性是否存在于数据中
        missing = [attr for attr in attributes if attr not in data.columns]
        if missing:
            raise ValueError(f"属性列不存在: {missing}")

        pos = 0
        grouped = data.groupby(attributes)

        self._log(f"按 {attributes} 分组得到 {len(grouped)} 个等价类")
        for name, group in grouped:
            decisions = group[decision_col].unique()
            if len(decisions) == 1:
                pos += len(group)
                self._log(f"{name}: {len(group)}样本 决策一致({decisions[0]})", indent=1)
            else:
                self._log(f"{name}: {len(group)}样本 决策冲突({decisions})", indent=1)

        dependency = pos / len(data)
        self._log(f"正区域样本: {pos}/{len(data)} ({dependency:.2%})")
        return dependency

    def compute_significance(self, data, core_attrs, decision_col):
        self._log("\n=== 属性重要性分析 ===")
        gamma_full = self.compute_dependency(data, core_attrs, decision_col)
        significance = {}

        for attr in core_attrs:
            reduced = [a for a in core_attrs if a != attr]
            if not reduced:
                gamma_reduced = 0.0
                self._log(f"移除 {attr} 后无剩余属性", indent=1)
            else:
                gamma_reduced = self.compute_dependency(data, reduced, decision_col)

            sig = gamma_full - gamma_reduced
            significance[attr] = sig
            self._log(f"{attr}: {gamma_full:.4f} - {gamma_reduced:.4f} = {sig:.4f}", indent=1)

        return significance

    def evaluate_model(self, data, features, target, cv=5):
        # 内部方法：执行交叉验证并记录结果
        def _run_evaluation(feature_set, set_name):
            # 调整cv的逻辑保持不变
            class_counts = data[target].value_counts()
            min_class_count = class_counts.min()
            adjusted_cv = max(min_class_count, 2) if min_class_count < cv else cv

            model = RandomForestClassifier(n_estimators=100)
            scores = cross_val_score(model, data[feature_set], data[target], cv=adjusted_cv)

            self._log(f"{set_name}各折准确率:")
            for i, score in enumerate(scores):
                self._log(f"折{i + 1}: {score:.2%}", indent=1)

            avg_score = scores.mean()
            self._log(f"{set_name}平均准确率: {avg_score:.2%}\n")
            return avg_score

        self._log("\n=== 模型验证 ===")

        # 先评估全特征集（排除目标列）
        all_features = [col for col in data.columns if col != target]
        full_score = _run_evaluation(all_features, "[全特征]")

        # 再评估核特征集
        core_score = _run_evaluation(features, "[核特征]")

        # 对比结果分析
        self._log("=== 准确率对比 ===")
        self._log(f"全特征属性数: {len(all_features)}", indent=1)
        self._log(f"核特征属性数: {len(features)}", indent=1)
        self._log(f"准确率变化: {core_score - full_score:+.2%}", indent=1)

        # 根据对比结果给出建议
        if core_score < full_score - 0.05:  # 核特征准确率低5%以上
            self._log("警告：核特征准确率显著低于全特征，建议检查：", indent=1)
            self._log("1. 核属性可能遗漏重要特征", indent=2)
            self._log("2. 数据预处理需要优化", indent=2)
        elif core_score > full_score:
            self._log("核特征表现优于全特征，约简有效！", indent=1)

        return core_score  # 保持原返回值不变

    def evaluate_algorithm(self, algo_func, *args):
        """带时间记录的评估"""
        start = time.time()
        result = algo_func(*args)
        elapsed = time.time() - start
        self.time_records[algo_func.__name__] = elapsed
        return result

    def get_comparison_metrics(self):
        """获取对比指标"""
        return {
            "accuracy": self.accuracy,
            "time_cost": self.time_records,
            "num_attributes": len(self.core_attrs)
        }