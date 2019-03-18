import numpy as np
from scipy.stats import expon, truncexpon, uniform


# Based on Mondrian Forests: Efficient Online Random Forests
# https://arxiv.org/pdf/1406.2673.pdf


GAMMA = 1


def data_ranges(data):
    return np.min(data, axis=0), np.max(data, axis=0)


class MondrianTree:
    def __init__(self, budget=np.inf):
        self.leaf_nodes = set()
        self.budget = budget
        self.classes = None
        self.class_indices = None
        self.root = None
        self.X = None
        self.y = None
        self.fitted = False

    # Algorithm 1 + fully online option
    def fit(self, X, y, online=False):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        if not online:
            self.root = MondrianBlock(X, y, parent=None, budget=self.budget, tree=self)
            self.compute_predictive_posterior()
        else:
            self.root = MondrianBlock(X[:2], y[:2], parent=None, budget=self.budget, tree=self)
            for i in range(2, len(y)):
                self.extend(X[i], y[i])

        self.fitted = True

    # Algorithm 7
    def compute_predictive_posterior(self):
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.parent is None:
                parent_posterior = np.ones_like(self.classes) / len(self.classes)  # H
            else:
                parent_posterior = node.parent.posterior_predictive

            class_counts = node.class_counts
            tables = node.tables
            discount = node.discount

            node.posterior_predictive = (class_counts - discount * tables
                                         + discount * np.sum(tables) * parent_posterior) / np.sum(class_counts)

            if node.left:
                queue = [node.left] + queue
            if node.right:
                queue = [node.right] + queue

    # Algorithm 8
    def predict(self, x):
        assert len(x.shape) == 1  # prediction for single x for now

        x += 1e-12  # dirty hack in case x is included in the training

        current = self.root
        pnsy = 1.
        s = np.zeros_like(self.classes, dtype=np.float64)
        while True:
            cost_difference = current.cost - current._parent_cost()
            eta = (np.maximum(x - current.upper, 0) + np.maximum(current.lower - x, 0)).sum()
            psjx = -np.expm1(-eta * cost_difference)
            if psjx > 0:
                expected_discount = (eta / (eta + GAMMA)) * (-np.expm1(-(eta + GAMMA) * cost_difference)) \
                           / (-np.expm1(-eta * cost_difference))

                class_counts = tables = np.minimum(current.class_counts, 1)

                if current.parent is None:
                    tilde_parent_posterior = np.ones_like(self.classes) / len(self.classes)
                else:
                    tilde_parent_posterior = current.parent.posterior_predictive

                posterior = (class_counts / np.sum(class_counts) - expected_discount * tables
                             + expected_discount * tables.sum() * tilde_parent_posterior)
                s += pnsy * psjx * posterior
            if current.is_leaf:
                s += pnsy * (1 - psjx) * current.posterior_predictive
                return s
            else:
                pnsy *= 1 - psjx
                if x[current.delta] <= current.xi:
                    current = current.left
                else:
                    current = current.right

    def extend(self, X, y):
        self.root.extend(X, y)


class MondrianBlock:
    def __init__(self, X, y, budget, parent=None, tree: MondrianTree = None, fit=True):
        assert tree
        self.tree = tree
        self.parent = parent
        self.left = None
        self.right = None
        self.budget = budget
        self.discount = 0
        self.lower = np.zeros(X.shape[1]) if X is not None else None
        self.upper = np.zeros_like(self.lower) if X is not None else None
        self.sides = np.zeros_like(self.lower) if X is not None else None
        self.class_counts = np.zeros_like(self.tree.classes)  # not exactly _counts_
        self.tables = np.zeros_like(self.tree.classes)  # see Chinese restaurants notation in the paper
        self.is_leaf = True  # will be set to False when needed

        if fit:
            self._fit(X, y)

    def _parent_cost(self):
        if self.parent is None:
            return 0.
        else:
            return self.parent.cost

    # Algorithm 5
    def _initialize_posterior_counts(self, X, y):
        for i, cls in enumerate(self.tree.classes):
            self.class_counts[i] = np.count_nonzero(y == cls)
        current = self
        while True:
            if not current.is_leaf:
                l_tables = current.left.tables if current.left else np.zeros_like(current.class_counts)
                r_tables = current.right.tables if current.right else np.zeros_like(current.class_counts)
                current.class_counts = l_tables + r_tables
            current.tables = np.minimum(current.class_counts, 1)
            if current.parent is None:
                break
            else:
                current = current.parent

    # Algorithm 6
    def _update_posterior_counts(self, y):
        class_index = self.tree.class_indices[y]
        self.class_counts[class_index] += 1
        current = self
        while True:
            if current.tables[class_index] == 1:
                return
            else:
                if not current.is_leaf:
                    l_table = current.left.tables[class_index] if current.left else 0
                    r_table = current.right.tables[class_index] if current.right else 0
                    current.class_counts[class_index] = l_table + r_table
                current.tables[class_index] = np.minimum(current.class_counts[class_index], 1)
                if current.parent is None:
                    return
                else:
                    current = current.parent

    # Algorithm 9
    def _fit(self, X, y):
        self.lower, self.upper = data_ranges(X)
        self.sides = self.upper - self.lower

        if len(y) <= 0 or np.all(y == y[0]):  # all labels identical
            self.cost = self.budget
        else:
            split_cost = expon.rvs(scale=(1 / self.sides.sum()))
            self.cost = self._parent_cost() + split_cost

        if self.cost < self.budget:
            # choose split dimension delta and location xi
            self.delta = np.random.choice(np.arange(X.shape[1]), p=(self.sides / self.sides.sum()))
            self.xi = uniform.rvs(loc=self.lower[self.delta], scale=self.sides[self.delta])

            # perform an actual split
            left_indices = X[:, self.delta] <= self.xi
            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[~left_indices], y[~left_indices]

            # sample children
            self.is_leaf = False
            # we first create unfitted blocks and then fit because otherwise self.left and self.right
            # may be accessed in ._initialize_posterior_counts before being assigned
            self.left = MondrianBlock(X_left, y_left, budget=self.budget, parent=self, tree=self.tree, fit=False)
            self.left._fit(X_left, y_left)
            self.right = MondrianBlock(X_right, y_right, budget=self.budget, parent=self, tree=self.tree, fit=False)
            self.right._fit(X_right, y_right)
        else:
            self.cost = self.budget
            self.tree.leaf_nodes.add(self)
            self._initialize_posterior_counts(X, y)

        self.discount = np.exp(-GAMMA * (self.cost - self._parent_cost()))

    def _get_subset_indices(self):
        return np.all(self.tree.X >= self.lower, axis=1) & np.all(self.tree.X <= self.upper, axis=1)

    def _get_label_subset(self, indices=None):
        if indices is None:
            indices = self._get_subset_indices()
        return self.tree.y[indices]

    def _get_feature_subset(self, indices=None):
        if indices is None:
            indices = self._get_subset_indices()
        return self.tree.X[indices]

    def _get_feature_label_subset(self, indices=None):
        if indices is None:
            indices = self._get_subset_indices()
        return self._get_feature_subset(indices), self._get_label_subset(indices)

    # Algorithm 10
    def extend(self, x, y):
        labels = self._get_label_subset()
        if len(labels) <= 0 or np.all(labels == labels[0]):  # all labels identical
            self.lower = np.minimum(self.lower, x)
            self.upper = np.maximum(self.upper, x)
            self.tree.X = np.vstack((self.tree.X, x))  # TODO: we possibly don't have to
            self.tree.y = np.hstack((self.tree.y, y))
            if y == labels[0]:
                self._update_posterior_counts(y)
                return
            else:
                self.tree.leaf_nodes.remove(self)
                X, y = self._get_feature_label_subset()
                self._fit(X, y)
        else:
            el = np.maximum(self.lower - x, np.zeros_like(x))
            eu = np.maximum(x - self.upper, np.zeros_like(x))
            sum_e = el + eu

            split_cost = expon.rvs(scale=(1 / sum_e.sum()))
            if self._parent_cost() + split_cost < self.budget:
                delta = np.random.choice(np.arange(len(x)), p=(sum_e / sum_e.sum()))
                if x[delta] > self.upper[delta]:
                    xi = uniform.rvs(loc=self.upper[delta], scale=x[delta] - self.upper[delta])
                else:
                    xi = uniform.rvs(loc=x[delta], scale=self.lower[delta] - x[delta])

                j_tilde = MondrianBlock(None, None, budget=self.budget, parent=self.parent, tree=self.tree, fit=False)
                j_tilde_attrs = {
                    'delta': delta,
                    'xi': xi,
                    'cost': self._parent_cost() + split_cost,
                    'lower': np.minimum(self.lower, x),
                    'upper': np.maximum(self.upper, x),
                    'sides': np.maximum(self.upper, x) - np.minimum(self.lower, x),
                    'is_leaf': False,
                }
                for attr, value in j_tilde_attrs.items():
                    setattr(j_tilde, attr, value)

                if self.parent is None:
                    self.tree.root = j_tilde
                else:
                    if self is self.parent.left:
                        self.parent.left = j_tilde
                    elif self is self.parent.right:
                        self.parent.right = j_tilde

                j_primes = MondrianBlock(X=np.array([x]), y=np.array([y]), budget=self.budget,
                                         parent=j_tilde, tree=self.tree)
                if x[delta] > xi:
                    j_tilde.left = self
                    j_tilde.right = j_primes
                else:
                    j_tilde.left = j_primes
                    j_tilde.right = self
            else:
                self.lower = np.minimum(self.lower, x)
                self.upper = np.maximum(self.upper, x)
                if not self.is_leaf:
                    if x[self.delta] <= self.xi:
                        child = self.left
                    else:
                        child = self.right
                    child.extend(x, y)

    def update_predictive_posterior(self):
        pass
