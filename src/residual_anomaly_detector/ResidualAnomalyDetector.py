from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

VERBOSE = True


class ResidualAnomalyDetector:
    classifiers = dict(dt=DecisionTreeClassifier)
    regressors = dict(dt=DecisionTreeRegressor)

    def __init__(
        self,
        classifier="dt",
        regressor="dt",
        clf_kwargs=dict(),
        rgr_kwargs=dict(),
        **algorithm_kwargs
    ):
        # Algorithms
        self.regressor_algorithm = regressor
        self.classifier_algorithm = classifier
        self.classifier_config = {**clf_kwargs, **algorithm_kwargs}
        self.regressor_config = {**rgr_kwargs, **algorithm_kwargs}
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

        self._init_models(X)
        self._fit_models(X)

        return

    def predict(X):
        raise NotImplementedError("Nope.")

    # Private Methods
    def _init_models(self, X):
        self.models = [
            self.classifier_algorithm(**self.classifier_config)
            if a in self.nominal_ids
            else self.regressor_algorithm(**self.regressor_config)
            for a in self.attr_ids
        ]
        return

    def _fit_models(self, X):
        for m_idx, desc_ids in range(self.n_models):
            self.models[m_idx].fit(X[:, self.desc_ids[m_idx]])
            
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

    @models.setter
    def models(self, value):
        assert isinstance(value, list)
        assert len(value) == self.n_attributes, "The amount of models must equal the amount of attributes"
        self._models = value
        return 

    @property
    def n_models(self):
        return len(self.models)

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

