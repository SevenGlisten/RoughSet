import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class ReductionAlgorithmBase:
    """属性约简算法基类"""

    def __init__(self, data, decision_col, log_callback):
        self.data = data
        self.decision_col = decision_col
        self.log = log_callback
        self.condition_cols = [col for col in data.columns if col != decision_col]

    def fitness(self, subset):
        """评估函数：准确率 - 属性数量惩罚"""
        if len(subset) == 0:
            return 0.0

        X = self.data[subset]
        y = self.data[self.decision_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        # 平衡准确率和属性数量（惩罚系数可调）
        return acc * 0.8 + (1 - len(subset) / len(self.condition_cols)) * 0.2


class GeneticAlgorithmReducer(ReductionAlgorithmBase):
    """遗传算法属性约简"""

    def __init__(self, data, decision_col, log_callback,
                 pop_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1):
        super().__init__(data, decision_col, log_callback)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _init_population(self):
        """初始化种群：二进制编码"""
        return np.random.randint(2, size=(self.pop_size, len(self.condition_cols)))

    def _select(self, fitness):
        """轮盘赌选择"""
        prob = fitness / fitness.sum()
        return np.random.choice(
            np.arange(self.pop_size),
            size=self.pop_size,
            p=prob
        )

    def _crossover(self, parent1, parent2):
        """单点交叉"""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1) - 1)
            return np.hstack([parent1[:point], parent2[point:]])
        return parent1.copy()

    def _mutate(self, individual):
        """位翻转突变"""
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def run(self):
        """执行算法"""
        self.log(f"\n=== 遗传算法开始 (种群{self.pop_size}, 代数{self.generations}) ===")
        population = self._init_population()
        best_fitness = -np.inf
        best_subset = []

        for gen in range(self.generations):
            # 评估适应度
            fitness = np.array([self.fitness(self._decode(ind)) for ind in population])

            # 记录最佳
            current_best = fitness.argmax()
            if fitness[current_best] > best_fitness:
                best_fitness = fitness[current_best]
                best_subset = self._decode(population[current_best])

            # 选择-交叉-变异
            selected = self._select(fitness)
            new_pop = []
            for i in range(0, self.pop_size, 2):
                parent1 = population[selected[i]]
                parent2 = population[selected[i + 1]]
                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1)
                new_pop.extend([self._mutate(child1), self._mutate(child2)])
            population = np.array(new_pop)

            # 日志
            self.log(f"代数 {gen + 1}: 最佳适应度 {best_fitness:.4f}, 属性数 {len(best_subset)}")

        return best_subset

    def _decode(self, individual):
        """解码二进制到属性名"""
        return [self.condition_cols[i] for i in np.where(individual == 1)[0]]


class PSOReducer(ReductionAlgorithmBase):
    """粒子群优化算法属性约简"""

    def __init__(self, data, decision_col, log_callback,
                 n_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        super().__init__(data, decision_col, log_callback)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _sigmoid(self, x):
        """Sigmoid函数用于概率映射"""
        return 1 / (1 + np.exp(-x))

    def run(self):
        """执行算法"""
        self.log(f"\n=== PSO算法开始 (粒子数{self.n_particles}, 迭代{self.max_iter}) ===")
        dim = len(self.condition_cols)

        # 初始化粒子
        position = np.random.uniform(-1, 1, (self.n_particles, dim))
        velocity = np.zeros((self.n_particles, dim))
        best_pos = position.copy()
        best_fitness = np.array([self.fitness(self._decode(pos)) for pos in self._sigmoid(position)])
        global_best_idx = best_fitness.argmax()
        global_best = best_pos[global_best_idx]

        for iter in range(self.max_iter):
            for i in range(self.n_particles):
                # 更新速度
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (best_pos[i] - position[i]) +
                               self.c2 * r2 * (global_best - position[i]))

                # 更新位置
                position[i] += velocity[i]

                # 评估新位置
                current_fitness = self.fitness(self._decode(self._sigmoid(position[i])))

                # 更新个体和全局最优
                if current_fitness > best_fitness[i]:
                    best_pos[i] = position[i].copy()
                    best_fitness[i] = current_fitness
                    if current_fitness > best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best = best_pos[i].copy()

            # 日志
            self.log(f"迭代 {iter + 1}: 全局最佳适应度 {best_fitness[global_best_idx]:.4f}")

        return self._decode(self._sigmoid(global_best))

    def _decode(self, particle):
        """解码粒子位置到属性名"""
        return [self.condition_cols[i] for i in np.where(particle > 0.5)[0]]