import pandas as pd
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QFileDialog, QProgressBar,
                             QComboBox, QLabel, QSpinBox, QToolBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from data.preprocessing import DataPreprocessor
from core.algorithms import CoreAlgorithms
from core.advanced_algorithms import GeneticAlgorithmReducer, PSOReducer
from core.evaluation import Evaluator
import time

from log.log_exporter import LogExporter


class AnalysisThread(QThread):
    progress = pyqtSignal(int, str)  # (进度百分比, 消息)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data, decision_col, algorithm, ga_params=None, pso_params=None,n_bins=5,binning_strategy="quantile"):
        super().__init__()
        self.data = data
        self.decision_col = decision_col
        self.algorithm = algorithm
        self.ga_params = ga_params or {}
        self.pso_params = pso_params or {}
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy

    def run(self):
        try:
            start_time = time.time()
            results = {}
            processed_data = None

            # 公共预处理
            self.progress.emit(20, "[预处理] 正在处理数据...")
            preprocessor = DataPreprocessor(log_callback=lambda msg: self.progress.emit(-1, msg),n_bins=self.n_bins,binning_strategy=self.binning_strategy)
            processed_data = preprocessor.preprocess(self.data.copy())

            # 定义分析方法
            algorithms = {
                "hybrid": lambda: self._run_hybrid(processed_data),
                "ga": lambda: self._run_ga(processed_data),
                "pso": lambda: self._run_pso(processed_data)
            }

            # 执行选定算法
            if self.algorithm in algorithms:
                result = algorithms[self.algorithm]()
                results[self.algorithm] = result
            else:
                raise ValueError("未知的算法选择")

            # 统一结果处理
            self.progress.emit(100, "分析完成！")
            self.result.emit({
                "type": self.algorithm,
                "data": results,
                "time": time.time() - start_time
            })

        except Exception as e:
            self.error.emit(f"分析失败: {str(e)}")

    def _run_hybrid(self, data):
        """执行混合粗糙集方法"""
        self.progress.emit(50, "[核心算法] 正在计算核属性...")
        algo = CoreAlgorithms(log_callback=lambda msg: self.progress.emit(-1, msg))
        core = algo.hybrid_core(data, self.decision_col, 0.01)
        return self._evaluate_result(data, core)

    def _run_ga(self, data):
        """执行遗传算法"""
        self.progress.emit(50, "[核心算法] 正在运行遗传算法...")
        reducer = GeneticAlgorithmReducer(
            data=data,
            decision_col=self.decision_col,
            log_callback=lambda msg: self.progress.emit(-1, msg),
            **self.ga_params
        )
        core = reducer.run()
        return self._evaluate_result(data, core)

    def _run_pso(self, data):
        """执行粒子群算法"""
        self.progress.emit(50, "[核心算法] 正在运行粒子群算法...")
        reducer = PSOReducer(
            data=data,
            decision_col=self.decision_col,
            log_callback=lambda msg: self.progress.emit(-1, msg),
            **self.pso_params
        )
        core = reducer.run()
        return self._evaluate_result(data, core)

    def _evaluate_result(self, data, core):
        """统一评估结果"""
        evaluator = Evaluator(log_callback=lambda msg: self.progress.emit(-1, msg))
        return {
            "core": core,
            "dependency": evaluator.compute_dependency(data, core, self.decision_col),
            "significance": evaluator.compute_significance(data, core, self.decision_col),
            "accuracy": evaluator.evaluate_model(data, core, self.decision_col)
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.data = None
        self.current_params = {}

        # 正确初始化工具栏
        self.toolbar = QToolBar("主工具栏")  # 创建工具栏对象
        self.addToolBar(self.toolbar)  # 将工具栏添加到窗口
        # 修改导出按钮
        self.btn_export = QPushButton("导出日志")
        self.btn_export.clicked.connect(self.export_log)
        self.toolbar.addWidget(self.btn_export)

        # 添加分箱数调整控件
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(1, 100)
        self.bins_spinbox.setValue(5)
        self.toolbar.addWidget(QLabel("分箱数:"))
        self.toolbar.addWidget(self.bins_spinbox)

        # 添加分箱规则选择下拉框
        self.binning_strategy_selector = QComboBox()
        self.binning_strategy_selector.addItems([
            "等频分箱",
            "等宽分箱",
            "聚类分箱"
        ])
        self.toolbar.addWidget(QLabel("分箱规则:"))
        self.toolbar.addWidget(self.binning_strategy_selector)

    def init_ui(self):
        self.setWindowTitle("粗糙集分析工具 v2.0")
        self.setGeometry(100, 100, 1000, 800)

        # 主控件
        self.text_output = QTextEdit()
        self.btn_load = QPushButton("加载CSV")
        self.btn_analyze = QPushButton("开始分析")
        self.progress_bar = QProgressBar()

        # 算法选择
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems([
            "混合粗糙集方法",
            "遗传算法(GA)",
            "粒子群算法(PSO)"
        ])
        self.algorithm_selector.currentIndexChanged.connect(self._update_parameter_ui)

        # 参数配置区
        self.param_widget = QWidget()
        param_layout = QHBoxLayout()

        # 遗传算法参数
        self.ga_pop_size = QSpinBox()
        self.ga_pop_size.setRange(10, 500)
        self.ga_pop_size.setValue(50)
        param_layout.addWidget(QLabel("GA种群:"))
        param_layout.addWidget(self.ga_pop_size)

        self.ga_generations = QSpinBox()
        self.ga_generations.setRange(10, 1000)
        self.ga_generations.setValue(100)
        param_layout.addWidget(QLabel("GA代数:"))
        param_layout.addWidget(self.ga_generations)

        # 粒子群算法参数
        self.pso_particles = QSpinBox()
        self.pso_particles.setRange(10, 500)
        self.pso_particles.setValue(30)
        param_layout.addWidget(QLabel("PSO粒子:"))
        param_layout.addWidget(self.pso_particles)

        self.pso_iterations = QSpinBox()
        self.pso_iterations.setRange(10, 500)
        self.pso_iterations.setValue(100)
        param_layout.addWidget(QLabel("PSO迭代:"))
        param_layout.addWidget(self.pso_iterations)

        self.param_widget.setLayout(param_layout)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.btn_load)
        main_layout.addWidget(QLabel("选择算法:"))
        main_layout.addWidget(self.algorithm_selector)
        main_layout.addWidget(QLabel("算法参数:"))
        main_layout.addWidget(self.param_widget)
        main_layout.addWidget(self.btn_analyze)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.text_output)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 信号连接
        self.btn_load.clicked.connect(self.load_data)
        self.btn_analyze.clicked.connect(self.start_analysis)

        # 初始化UI状态
        self._update_parameter_ui()

    def export_log(self):
        """导出日志处理"""
        try:
            # 获取文本框内容
            log_content = self.text_output.toPlainText()

            # 调用导出模块
            filepath = LogExporter.export_to_txt(log_content)

            # 显示导出结果
            self.text_output.append(f"\n[系统] 日志已导出至：{filepath}")
        except Exception as e:
            self._show_error(str(e))

    def _update_parameter_ui(self):
        """根据选择的算法显示对应参数"""
        algo = self.algorithm_selector.currentText()
        show_ga = "遗传算法" in algo or "对比" in algo
        show_pso = "粒子群" in algo or "对比" in algo

        for i in range(self.param_widget.layout().count()):
            widget = self.param_widget.layout().itemAt(i).widget()
            if isinstance(widget, QSpinBox):
                label = self.param_widget.layout().itemAt(i - 1).widget()
                is_ga_param = "GA" in label.text()
                is_pso_param = "PSO" in label.text()

                widget.setVisible((is_ga_param and show_ga) or (is_pso_param and show_pso))
                label.setVisible((is_ga_param and show_ga) or (is_pso_param and show_pso))

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开CSV文件", "", "CSV文件 (*.csv)")
        if path:
            try:
                self.data = pd.read_csv(path)
                self.text_output.append(f"成功加载数据: {len(self.data)}行×{len(self.data.columns)}列")
                self.text_output.append(f"决策属性列: {self.data.columns[-1]}")
            except Exception as e:
                self._show_error(f"加载失败: {str(e)}")

    def start_analysis(self):
        if self.data is None:
            self._show_error("请先加载数据文件")
            return

        # 获取参数
        algo_map = {
            "混合粗糙集方法": "hybrid",
            "遗传算法(GA)": "ga",
            "粒子群算法(PSO)": "pso"
        }
        algo_type = algo_map[self.algorithm_selector.currentText()]

        # 配置参数
        ga_params = {
            "pop_size": self.ga_pop_size.value(),
            "generations": self.ga_generations.value()
        }

        pso_params = {
            "n_particles": self.pso_particles.value(),
            "max_iter": self.pso_iterations.value()
        }

        # 获取分箱数
        n_bins = self.bins_spinbox.value()

        # 获取分箱规则
        binning_strategy_map = {
            "等频分箱": "quantile",
            "等宽分箱": "uniform",
            "聚类分箱":"kmeans"
        }
        binning_strategy = binning_strategy_map[self.binning_strategy_selector.currentText()]

        # 创建分析线程
        self.thread = AnalysisThread(
            data=self.data,
            decision_col=self.data.columns[-1],
            algorithm=algo_type,
            ga_params=ga_params,
            pso_params=pso_params,
            n_bins=n_bins,
            binning_strategy=binning_strategy
        )

        # 连接信号
        self.thread.progress.connect(self.update_progress)
        self.thread.result.connect(self.handle_result)
        self.thread.error.connect(self._show_error)

        # 重置界面状态
        self.btn_analyze.setEnabled(False)
        self.text_output.clear()
        self.progress_bar.setValue(0)

        # 启动线程
        self.thread.start()

    def update_progress(self, percent, message):
        if percent >= 0:
            self.progress_bar.setValue(percent)
        self.text_output.append(message)
        # 自动滚动到底部
        self.text_output.verticalScrollBar().setValue(
            self.text_output.verticalScrollBar().maximum()
        )

    def handle_result(self, result):
        self.btn_analyze.setEnabled(True)
        data = result["data"][result["type"]]
        self.text_output.append("\n=== 最终结果 ===")
        self.text_output.append(f"分析耗时: {result['time']:.2f}秒")
        self.text_output.append(f"核属性 ({len(data['core'])}个): {data['core']}")
        self.text_output.append(f"依赖度: {data['dependency']:.2%}")
        self.text_output.append("属性重要性:")
        for attr, sig in data["significance"].items():
            self.text_output.append(f"  {attr}: {sig:.4f}")
        self.text_output.append(f"验证准确率: {data['accuracy']:.2%}")

    def _show_error(self, message):
        self.text_output.append(f'<font color="red">错误: {message}</font>')
        self.progress_bar.reset()
        self.btn_analyze.setEnabled(True)