from collections import defaultdict

import numpy as np
import pandas as pd

class CoreAlgorithms:
    def __init__(self, log_callback):  # 必须传入日志回调
        self.log = log_callback  # 绑定到图形界面
        self.step_details = []

    # === 差别矩阵法 ===
    def discernibility_matrix(self, data, decision_col):
        self._log("=== 改进版差别矩阵法 ===")
        condition_cols = [col for col in data.columns if col != decision_col]
        core = set()

        # 新增：日志限制相关变量
        diff_log_count = 0  # 已记录的差异日志数量
        max_diff_log = 100  # 最大允许记录数
        log_truncated = False  # 是否已触发截断

        # 数据结构优化
        diff_pairs = []  # 存储所有差异属性集合
        attr_counter = {col: 0 for col in condition_cols}  # 属性出现次数统计

        # 遍历时处理浮点数精度
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if data.iloc[i][decision_col] != data.iloc[j][decision_col]:
                    # 精度处理（与数据生成时的round(3)保持一致）
                    diff = [
                        col for col in condition_cols
                        if round(data.iloc[i][col], 3) != round(data.iloc[j][col], 3)
                    ]
                    if diff:
                        diff_pairs.append(diff)
                        # 统计属性出现次数
                        for col in diff:
                            attr_counter[col] += 1
                        # ========== 新增：限制日志输出 ==========
                        if diff_log_count < max_diff_log:
                            self._log(f"样本{i}-{j} 差异属性: {diff}", indent=1)
                            diff_log_count += 1
                        elif not log_truncated:  # 触发截断提示
                            self._log(f"(日志截断，避免日志过多，仅显示前{max_diff_log}条差异记录)", indent=1)
                            log_truncated = True
                        # ========== 限制结束 ==========

        # ========== 改进点1: 必要属性检测 ==========
        # 出现在所有差异对中的属性
        total_diff = len(diff_pairs)
        must_exist = [
            col for col, count in attr_counter.items()
            if count == total_diff and total_diff > 0
        ]
        if must_exist:
            self._log(f"必要属性: {must_exist} (出现在所有{total_diff}个差异对中)", indent=1)
            core.update(must_exist)

        # ========== 改进点2: 单属性差异检测 ==========
        single_attrs = [pair[0] for pair in diff_pairs if len(pair) == 1]
        if single_attrs:
            self._log(f"单属性差异对: {set(single_attrs)} (共{len(single_attrs)}对)", indent=1)
            core.update(single_attrs)

        # ========== 改进点3: 冗余属性过滤 ==========
        # 删除参与度低于50%的伪核属性
        final_core = []
        for col in core:
            participation_rate = attr_counter[col] / total_diff if total_diff > 0 else 0
            if participation_rate >= 0.5:  # 阈值可调
                final_core.append(col)
                self._log(f"保留属性 {col} (参与率: {participation_rate:.1%})", indent=1)
            else:
                self._log(f"过滤属性 {col} (参与率: {participation_rate:.1%})", indent=1)

        # ========== 日志统计 ==========
        self._log("\n属性参与度统计:")
        for col, cnt in sorted(attr_counter.items(), key=lambda x: -x[1]):
            rate = cnt / total_diff if total_diff > 0 else 0
            self._log(f"{col}: {cnt}/{total_diff} ({rate:.1%})", indent=1)

        return sorted(final_core)

    # === 信息熵法 ===
    def entropy_method(self, data, decision_col, threshold):
        self._log("\n=== 信息熵法属性约简 ===")
        self._log(f"决策属性列: {decision_col}", indent=1)
        self._log(f"信息增益阈值: {threshold}", indent=1)

        # === 计算总熵 ===
        total_entropy = self._calculate_entropy(data[decision_col])
        self._log(f"\n总熵 H(D) = {total_entropy:.4f}", indent=1)
        self._log("(值越大表示决策列不确定性越高)", indent=2)

        core = []
        condition_cols = [col for col in data.columns if col != decision_col]

        # === 遍历每个条件属性 ===
        for col in condition_cols:
            self._log(f"\n▶ 分析属性: {col}", indent=1)
            grouped = data.groupby(col)
            cond_entropy = 0.0
            total_samples = len(data)

            # === 分组计算 ===
            self._log(f"分组统计（共 {len(grouped)} 个不同值）:", indent=2)
            for name, group in grouped:
                group_size = len(group)
                prob = group_size / total_samples
                decisions = group[decision_col].value_counts().to_dict()

                # 计算当前组的熵
                e = self._calculate_entropy(group[decision_col])
                cond_entropy += prob * e

                # 详细分组日志
                self._log(f"值 '{name}':", indent=3)
                self._log(f"- 样本数: {group_size} (占比: {prob:.2f})", indent=4)
                self._log(f"- 决策分布: {decisions}", indent=4)
                self._log(f"- 分组熵 H(D|{col}={name}) = {e:.4f}", indent=4)

            # === 信息增益计算 ===
            info_gain = total_entropy - cond_entropy
            self._log(f"\n条件熵 H(D|{col}) = {cond_entropy:.4f}", indent=2)
            self._log(f"信息增益 IG({col}) = {total_entropy:.4f} - {cond_entropy:.4f} = {info_gain:.4f}", indent=2)

            # === 属性选择判断 ===
            if info_gain >= threshold:
                core.append(col)
                self._log(f"✅ 保留属性 {col} (IG ≥ {threshold})", indent=2)
            else:
                self._log(f"⭕ 丢弃属性 {col} (IG < {threshold})", indent=2)

        # === 最终结果 ===
        self._log(f"\n核心属性集合: {core}", indent=1)
        return core

    def _calculate_entropy(self, series):
        counts = series.value_counts(normalize=True)
        if len(counts) < 2:
            return 0.0
        entropy = -np.sum(counts * np.log2(counts + 1e-10))  # 防止log(0)
        return entropy

    def positive_region_reduction(self, data, decision_col, precision=3):
        """
        正区域法属性约简（详细日志版）
        :param data: 输入数据DataFrame
        :param decision_col: 决策属性名
        :param precision: 属性值比较精度(小数位数)
        :return: 核属性列表
        """
        self._log(f"\n=== 正区域法约简开始（比较精度={precision}位小数） ===")
        condition_cols = [col for col in data.columns if col != decision_col]
        core = set()

        # 计算全属性正区域
        self._log("步骤1: 计算全属性正区域", indent=1)
        full_region = self._calculate_positive_region(data, condition_cols, decision_col, precision)
        self._log(f"初始正区域大小: {len(full_region)}/{len(data)} (覆盖率: {len(full_region) / len(data):.1%})",
                  indent=2)

        # 逐个属性检测必要性
        self._log(f"\n步骤2: 必要性检测（共检测 {len(condition_cols)} 个属性）", indent=1)
        for idx, col in enumerate(condition_cols, 1):
            self._log(f"({idx}/{len(condition_cols)}) 检测属性 [{col}]：", indent=1)

            # 创建临时属性集
            reduced_cols = [c for c in condition_cols if c != col]
            self._log(f"移除 [{col}] 后的属性集: {reduced_cols}", indent=2)

            # 计算约简后正区域
            reduced_region = self._calculate_positive_region(data, reduced_cols, decision_col, precision)
            reduction = len(full_region) - len(reduced_region)

            # 判断正区域是否缩小
            if reduction > 0:
                core.add(col)
                self._log(f"→ 必要属性！正区域减少 {reduction} 个样本", indent=2)
                self._log(
                    f"  新正区域: {len(reduced_region)}/{len(data)} (减少比例: {reduction / len(full_region):.1%})",
                    indent=3)
            else:
                self._log(f"→ 冗余属性。正区域未变化", indent=2)

        # 最终结果
        self._log(f"\n步骤3: 核属性确定", indent=1)
        self._log(f"必要属性集合: {sorted(core)} (共{len(core)}个)", indent=2)
        return sorted(core)

    def _calculate_positive_region(self, data, condition_cols, decision_col, precision):
        """
        计算正区域（带详细等价类分析）
        :return: 正区域样本索引集合
        """
        equivalence_classes = defaultdict(set)
        self._log(f"生成等价类（属性集: {condition_cols})", indent=3)

        # 构建等价类
        for idx, row in data.iterrows():
            # 按精度生成特征键
            key = tuple(round(row[col], precision) for col in condition_cols)
            equivalence_classes[key].add(idx)

        # 输出等价类详情
        self._log(f"共生成 {len(equivalence_classes)} 个等价类：", indent=3)
        for key, samples in equivalence_classes.items():
            key_str = ", ".join(f"{k:.{precision}f}" for k in key)
            decisions = data.loc[list(samples), decision_col].unique()
            status = "正区域" if len(decisions) == 1 else "冲突区域"
            self._log(f"· 特征键 [{key_str}]: {len(samples)}样本 | 决策值: {decisions} → {status}", indent=4)

        # 筛选正区域
        positive_samples = set()
        for eq_class in equivalence_classes.values():
            decisions = data.loc[list(eq_class), decision_col].unique()
            if len(decisions) == 1:
                positive_samples.update(eq_class)

        return positive_samples


    def hybrid_core(self, data, decision_col, entropy_threshold):
        """混合方法详细步骤输出"""
        self.log("\n=== 混合方法筛选核属性 ===")

        # 调用差别矩阵法
        core_matrix = self.discernibility_matrix(data, decision_col)

        # 调用信息熵法
        core_entropy = self.entropy_method(data, decision_col, entropy_threshold)

        core_positive = self.positive_region_reduction(data, decision_col, precision=3)

        # 计算交集
        final_core = list(set(core_matrix) & set(core_entropy) & set(core_positive))

        # 检查核属性是否为空
        if not final_core:
            self.log("警告: 未找到核属性，使用全属性集")
            final_core = [col for col in data.columns if col != decision_col]

        self.log(
            f"\n交集操作:\n"
            f"  差别矩阵法候选: {core_matrix}\n"
            f"  信息熵法候选: {core_entropy}\n"
            f"  正区域法候选: {core_positive}\n"
            f"  最终核属性: {final_core}"
        )
        return final_core

    def _log(self, message, indent=0):
        """通过回调输出到界面"""
        prefix = "|  " * indent
        full_msg = f"{prefix}{message}"
        self.step_details.append(full_msg)
        self.log(full_msg)  # 调用图形界面的日志方法