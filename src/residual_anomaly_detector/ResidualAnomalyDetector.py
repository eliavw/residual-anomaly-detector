from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

VERBOSE = True


class ResidualAnomalyDetector:
    classifiers = dict(dt=DecisionTreeClassifier)
    regressors = dict(dt=DecisionTreeRegressor)

    def __init__(self, classifier="dt", regressor="dt"):
        # Algorithms
        self.regressor_algorithm = regressor
        self.classifier_algorithm = classifier
        self._desc_ids = None
        self._targ_ids = None

        self._models = None
        self._residuals = None
        self._scores = None
        self._attr_ids = None

        self.n_instances = None
        self.n_attributes = None
        return

    def fit(self, X, nominal_ids=None):
        self.n_instances, self.n_attributes = X.shape
        self.attr_ids = self.n_attributes
        self.nominal_ids = nominal_ids

        return

    def predict(X):
        raise NotImplementedError("Nope.")

    # Private Methods
    def _init_models(self, X):
        self._models = []
        return

    def _fit_models(self, X):
        self._models = [
            m.fit(X[:, desc_ids]) for m, desc_ids in zip(self.models, self.desc_ids)
        ]
        return

    def _predict_models(self, X_true):
        X_pred = None
        return

    def _get_residuals(self, X_true, X_pred):
        self._residuals = X_true - X_pred
        return

    def _get_scores(self):
        return

    # Properties
    @property
    def models(self):
        return self._models

    @property
    def residuals(self):
        return self._residuals

    @property
    def decision_scores_(self):
        return self.scores

    @property
    def scores(self):
        return self._scores

    @property
    def classifier_algorithm(self):
        return self._classifier_algorithm

    @property
    def regressor_algorithm(self):
        return self._regressor_algorithm

    @classifier_algorithm.setter
    def classifier_algorithm(self, value):
        self._classifier_algorithm = self.classifiers[value]
        return

    @regressor_algorithm.setter
    def regressor_algorithm(self, value):
        self._regressor_algorithm = self.regressors[value]
        return

    @property
    def desc_ids(self):
        return self._desc_ids

    @property
    def targ_ids(self):
        return self._targ_ids

    @property
    def attr_ids(self):
        return self._attr_ids

    @attr_ids.setter
    def attr_ids(self, n):
        self._attr_ids = set(range(n))
        return 

    @property
    def nominal_ids(self):
        return self._nominal_ids

    @nominal_ids.setter
    def nominal_ids(self, value):
        if value is None:
            self._nominal_ids = set()
        else:
            self._nominal_ids = set(value)
        return 

    @property
    def numeric_ids(self):
        return self.attr_ids - self.nominal_ids

