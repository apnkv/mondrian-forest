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
        self.root = None

    # Algorithm 1
    def fit(self, X, y=None):
        self.root = MondrianBlock(X, None, self.budget, self)

    # Algorithm 3
    def extend(self, x, y=None):
        self.root.extend(x, y)


class MondrianBlock:
    # tree pointer is needed only to add block to leaves
    def __init__(self, data, parent, budget, tree: MondrianTree = None, fit=True):
        assert tree
        self.tree = tree

        self.parent = parent
        self.left = None
        self.right = None
        self.is_leaf = True
        self.budget = budget
        self.cost = 0 if not parent else parent.cost

        self.delta = None
        self.xi = None

        if fit:
            self.lower, self.upper = data_ranges(data)
            self.sides = self.upper - self.lower
            self._fit(data)

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
        else:
            self.lower = np.minimum(self.lower, x)
            self.upper = np.maximum(self.upper, x)
            if self not in self.tree.leaf_nodes:
                if x[self.delta] <= self.xi:
                    child = self.left
                else:
                    child = self.right
                child.extend(x, y)


def plot_mondrian_tree(tree):
    pass
