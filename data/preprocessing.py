import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class DataPreprocessor:
    def __init__(self, log_callback, n_bins=5,binning_strategy="quantile"):  # 增加日志回调参数
        self.n_bins = n_bins
        self.discretizer = None
        self.log = log_callback  # 使用回调函数记录日志
        self.binning_strategy = binning_strategy

    def _log_step(self, message):
        """通过回调函数记录日志"""
        self.log(f"[预处理] {message}")

    def load_data(self, file_path):
        """加载数据"""
        self._log_step(f"正在加载文件: {file_path}")
        data = pd.read_csv(file_path)
        self._log_step(f"原始数据维度: {data.shape}")
        self._log_step(f"原始列名: {list(data.columns)}")
        return data

    def preprocess(self, data):
        """完整预处理流程"""
        # 步骤0: 记录当前配置
        self._log_step(f"当前分箱数配置: {self.n_bins}，分箱规则:{self.binning_strategy}")
        # 步骤1: 类型转换
        self._log_step("开始类型转换...")
        data = data.apply(pd.to_numeric, errors='coerce')
        failed_cols = data.columns[data.isnull().any()].tolist()
        if failed_cols:
            self._log_step(f"以下列包含非数值数据: {failed_cols}")

        # 步骤2: 缺失值处理
        self._log_step("处理缺失值...")
        original_count = len(data)  # 记录原始数据量
        data = data.dropna()  # 删除包含缺失值的行
        removed_count = original_count - len(data)
        self._log_step(
            f"删除包含缺失值的行 | 原始行数: {original_count} | 删除行数: {removed_count} | 剩余行数: {len(data)}")

        # 步骤3: 智能离散化
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.continuous_cols = [col for col in numeric_cols if col != data.columns[-1]]

        cols_to_discretize = []
        cols_to_keep = []

        # 判断哪些属性需要分箱
        for col in self.continuous_cols:
            unique_values = data[col].unique()
            if len(unique_values) > 10:
                cols_to_discretize.append(col)
            else:
                cols_to_keep.append(col)

        if cols_to_discretize:
            self._log_step(f"正在处理连续属性: {cols_to_discretize}")

            # 新增逻辑：根据样本量判断是否进行分箱
            n_samples = len(data)
            min_samples_per_bin = 5  # 每个分箱至少包含5个样本
            disable_discretization = False  # 控制是否完全禁用分箱

            # 逻辑1: 总样本量过少时完全跳过分箱
            if n_samples < 30:  # 样本量<30时不建议分箱
                self._log_step(f"样本量{n_samples}过少，跳过分箱操作")
                disable_discretization = True
            else:
                # 逻辑2: 动态计算最大允许分箱数
                max_possible_bins = max(1, n_samples // min_samples_per_bin)
                actual_bins = min(self.n_bins, max_possible_bins)

                if actual_bins < self.n_bins:
                    self._log_step(f"根据样本量自动调整分箱数: {self.n_bins} → {actual_bins}")
                    self.n_bins = actual_bins

            # 执行分箱 (仅在启用时)
            if not disable_discretization:
                try:
                    self.discretizer = KBinsDiscretizer(
                        n_bins=self.n_bins,
                        encode='ordinal',
                        strategy=self.binning_strategy
                    )
                    data[cols_to_discretize] = self.discretizer.fit_transform(data[cols_to_discretize])
                    data[cols_to_discretize] = data[cols_to_discretize].astype(int)

                    # 记录分箱边界
                    bin_edges = self.discretizer.bin_edges_
                    for idx, col in enumerate(cols_to_discretize):
                        edges = np.round(bin_edges[idx], 4)
                        self._log_step(f"{col} 分箱区间: {edges}")
                except ValueError as e:
                    # 处理quantile策略在小样本时的报错
                    self._log_step(f"分箱失败({str(e)})，回退到等宽分箱")
                    self.discretizer = KBinsDiscretizer(
                        n_bins=self.n_bins,
                        encode='ordinal',
                        strategy='uniform'  # 回退到等宽分箱
                    )
                    data[cols_to_discretize] = self.discretizer.fit_transform(data[cols_to_discretize])
                    data[cols_to_discretize] = data[cols_to_discretize].astype(int)
            else:
                # 保持原始连续值(可选：转换为整型)
                data[cols_to_discretize] = data[cols_to_discretize].astype(int)
                self._log_step(f"连续属性保持原始整型值: {cols_to_discretize}")

        if cols_to_keep:
            self._log_step(f"属性 {cols_to_keep} 不同值出现次数未达10次，保持原始值")

        # 步骤4: 决策列处理
        decision_col = data.columns[-1]
        data[decision_col] = data[decision_col].astype(int)
        self._log_step(f"决策列 '{decision_col}' 已转换为整型")

        return data