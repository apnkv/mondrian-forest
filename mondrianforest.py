import numpy as np
from scipy.stats import expon, uniform

# Based on Mondrian Forests: Efficient Online Random Forests
# https://arxiv.org/pdf/1406.2673.pdf


def data_ranges(data):
    """
    Computes l^x_{...} and u^x_{...}.
    :param data: input matrix
    :return: minimums and maximums by feature
    """
    return np.min(data, axis=0), np.max(data, axis=0)


class MondrianTree:
    def __init__(self, budget=np.inf):
        self.leaf_nodes = set()
        self.budget = budget
        self.classes = None
        self.class_indices = None
        self.regions = 0
        self.root = None

    # Algorithm 1 + online option
    def fit(self, X, y, simulate_online=False):
        self.classes = np.unique(y)
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        if not simulate_online:
            self.root = MondrianBlock(X, None, self.budget, self)
            self.batch_initialize_posterior_counts(X, y)
            self.compute_predictive_posterior()
        else:
            self.root = MondrianBlock(X[:2], None, self.budget, self)
            self.batch_initialize_posterior_counts(X[:2], y[:2])
            for i in range(2, len(X)):
                self.extend(X[i], y[i])

    # Algorithm 3
    def extend(self, x, y):
        x_leaf = self.root.extend(x, y)
        x_leaf.update_posterior_counts(y)

    # Algorithm 5
    def batch_initialize_posterior_counts(self, X, y):
        queue = list(self.leaf_nodes)
        while queue:
            node = queue.pop()
            print('init posterior,', node)
            if not node.posterior_initialized:
                node_indices = np.all(X >= node.lower, axis=1) & np.all(X <= node.upper, axis=1)
                labels_in_node = y[node_indices]
                label_distribution = np.zeros_like(self.classes)

                if node.is_leaf:
                    for i, cls in enumerate(self.classes):
                        label_distribution[i] = np.count_nonzero(labels_in_node == cls)
                else:
                    label_distribution += node.left.label_distribution if node.left else 0
                    label_distribution += node.right.label_distribution if node.right else 0

                node.label_distribution = label_distribution
                node.tables = np.minimum(label_distribution, 1)
                node.posterior_initialized = True
                if not node.parent:
                    break
                else:
                    queue = [node.parent] + queue

    # Algotithm 7
    def compute_predictive_posterior(self):
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node is self.root:
                parent_posterior = np.ones_like(self.classes) / len(self.classes)  # H
            else:
                parent_posterior = node.parent.posterior_predictive

            node_c = node.label_distribution
            node_tab = node.tables
            discount = node.discount

            node.posterior_predictive = (node_c - discount * node_tab
                                         + discount * np.sum(node_tab) * parent_posterior) / np.sum(node_c)

            if node.left:
                queue = [node.left] + queue
            if node.right:
                queue = [node.right] + queue

    def leaf(self, x):
        node = self.root
        while not node.is_leaf:
            if any(x < node.lower) or any(x > node.upper):
                return None
            if x[node.delta] <= node.delta:
                node = node.left
            else:
                node = node.right
        return node


class MondrianBlock:
    # tree pointer is needed only to add block to leaves
    def __init__(self, data, parent, budget, tree: MondrianTree = None, fit=True):
        assert tree
        self.tree = tree
        self.number = tree.regions
        tree.regions += 1

        self.parent = parent
        self.left = None
        self.right = None
        self.is_leaf = True
        self.budget = budget
        self.cost = 0 if not parent else parent.cost
        self.discount = 0

        self.label_distribution = np.zeros_like(tree.classes)
        self.tables = np.zeros_like(tree.classes)  # "Chinese restaurants" notation from the paper
        self.posterior_predictive = np.zeros_like(tree.classes)
        self.posterior_initialized = False

        self.delta = None
        self.xi = None

        if fit:
            self.lower, self.upper = data_ranges(data)
            self.sides = self.upper - self.lower
            self._fit(data)

    def __repr__(self):
        return f'MB {self.number}{" (l)" if self.is_leaf else ""}'

    # Algorithm 2
    def _fit(self, data):
        # one point in block
        if self.sides.sum() <= 0:
            self.is_leaf = True
            self.tree.leaf_nodes.add(self)
            return

        split_cost = expon.rvs(scale=(1 / self.sides.sum()))
        parent_cost = self.parent.cost if self.parent else 0.
        if parent_cost + split_cost < self.budget:
            # choose split dimension delta and location xi
            delta = np.random.choice(np.arange(data.shape[1]), p=(self.sides / self.sides.sum()))
            xi = uniform.rvs(loc=self.lower[delta], scale=self.sides[delta])

            # save split parameters
            self.cost = parent_cost + split_cost
            self.delta = delta
            self.xi = xi

            # perform an actual split
            data_left = data[data[:, delta] <= xi]
            data_right = data[data[:, delta] > xi]

            # sample children
            self.left = MondrianBlock(data_left, self, self.budget, self.tree)
            self.right = MondrianBlock(data_right, self, self.budget, self.tree)
            self.is_leaf = False
        else:
            self.cost = self.budget
            self.is_leaf = True
            self.tree.leaf_nodes.add(self)

        self.discount = np.exp(-(self.cost - parent_cost))

    # Algorithm 4
    def extend(self, x, y=None):
        el = np.maximum(self.lower - x, np.zeros_like(x))
        eu = np.maximum(x - self.upper, np.zeros_like(x))
        sum_e = el + eu

        split_cost = expon.rvs(scale=(1 / sum_e.sum()))
        parent_cost = self.parent.cost if self.parent else 0.
        if parent_cost + split_cost < self.cost:
            delta = np.random.choice(np.arange(len(x)), p=(sum_e / sum_e.sum()))
            if x[delta] > self.upper[delta]:
                xi = uniform.rvs(loc=self.upper[delta], scale=x[delta]-self.upper[delta])
            else:
                xi = uniform.rvs(loc=x[delta], scale=self.lower[delta]-x[delta])

            j_tilde = MondrianBlock(
                data=None,
                parent=self.parent.parent if self.parent else None,
                budget=self.budget,
                tree=self.tree,
                fit=False
            )
            j_tilde_attrs = {
                'delta': delta,
                'xi': xi,
                'cost': parent_cost + split_cost,
                'lower': np.minimum(self.lower, x),
                'upper': np.maximum(self.upper, x),
                'sides': np.maximum(self.upper, x) - np.minimum(self.lower, x)
            }

            for attr, value in j_tilde_attrs.items():
                setattr(j_tilde, attr, value)

            j_primes = MondrianBlock(np.array([x]), j_tilde, self.budget, self.tree)
            if x[delta] <= xi:
                j_tilde.left = j_primes
                j_tilde.right = self
            else:
                j_tilde.left = self
                j_tilde.right = j_primes

            if self.parent:
                if self.parent.left is self:
                    self.parent.left = j_tilde
                elif self.parent.right is self:
                    self.parent.right = j_tilde

            self.parent = j_tilde
            if not j_tilde.parent:
                self.tree.root = j_tilde

            return j_primes  # ought to be the leaf of x
        else:
            self.lower = np.minimum(self.lower, x)
            self.upper = np.maximum(self.upper, x)
            if self not in self.tree.leaf_nodes:
                if x[self.delta] <= self.xi:
                    child = self.left
                else:
                    child = self.right
                return child.extend(x, y)

    # Algorithm 6
    def update_posterior_counts(self, y):
        label_index = self.tree.class_indices[y]
        self.label_distribution[label_index] += 1
        current = self
        while True:
            if self.tables[label_index] == 1:
                return
            else:
                if not current.is_leaf:
                    current.label_distribution[label_index] = 0
                    current.label_distribution[label_index] += self.left.tables[label_index] if self.left else 0
                    current.label_distribution[label_index] += self.right.tables[label_index] if self.right else 0
                    current.tables[label_index] = min(current.label_distribution[label_index], 1)
                    if current is self.tree.root:
                        return
                    else:
                        current = current.parent


def plot_mondrian_tree(tree):
    pass
