from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from scipy.stats import t

VERBOSE = True


class ResidualAnomalyDetector:
    classifiers = dict(dt=DecisionTreeClassifier)
    regressors = dict(dt=DecisionTreeRegressor)

    def __init__(
        self,
        classifier="dt",
        regressor="dt",
        significance_level=0.05,
        clf_kwargs=dict(),
        rgr_kwargs=dict(),
        verbose=VERBOSE,
        **algorithm_kwargs
    ):
        # General params
        self.verbose = verbose

        # Metadata
        self.n_instances = None
        self.n_attributes = None
        self._attr_ids = None
        self._nominal_ids = None

        # Models
        self._models = None
        self._desc_ids = None
        self._targ_ids = None

        self.regressor_algorithm = regressor
        self.classifier_algorithm = classifier
        self.classifier_config = {**clf_kwargs, **algorithm_kwargs}
        self.regressor_config = {**rgr_kwargs, **algorithm_kwargs}

        # Residuals/Anomaly Scores
        self.significance_level = significance_level
        self._residuals = None
        self._scores = None
        self._labels = None

        return

    def fit(self, X, nominal_ids=None):
        self.n_instances, self.n_attributes = X.shape
        self.attr_ids = self.n_attributes
        self.nominal_ids = nominal_ids

        # Fit Models
        self._init_desc_and_targ_ids()
        self._init_models(X)
        self._fit_models(X)

        # Get Scores
        self._labels = self._init_labels()
        self._residuals = self._get_residuals(X)
        self._scores = self._get_scores()

        return

    def predict(self):
        progress_to_be_made = True
        while progress_to_be_made:
            n_anomalies_start = self.n_anomalies
            outlier_idxs = self._detect_outliers()
            self._update_labels(outlier_idxs)
            n_anomalies_after = self.n_anomalies
            progress_to_be_made = n_anomalies_after > n_anomalies_start

            if self.verbose:
                msg = """
                In this iteration, I found anomalies: {}
                Total n_anomalies now: {}
                """.format(
                    outlier_idxs, self.n_anomalies
                )
                print(msg)
        return self.labels

    @staticmethod
    def normalize_residuals(residuals):
        avg_residuals = np.mean(residuals, axis=0)
        std_residuals = np.std(residuals, axis=0)
        R = (np.abs(residuals - avg_residuals)) / std_residuals
        return R

    # Private Methods
    def _init_desc_and_targ_ids(self):
        d = []
        t = []
        for a in self.attr_ids:
            d.append(self.attr_ids - {a})
            t.append({a})

        self._desc_ids = d
        self._targ_ids = t

        return

    def _init_models(self, X):
        self.models = [
            self.classifier_algorithm(**self.classifier_config)
            if a in self.nominal_ids
            else self.regressor_algorithm(**self.regressor_config)
            for a in self.attr_ids
        ]
        return

    def _init_labels(self):
        # Initialize everything normal
        return np.zeros(self.n_instances)

    def _fit_models(self, X):
        for m_idx in range(self.n_models):
            desc_ids = list(self.desc_ids[m_idx])
            targ_ids = list(self.targ_ids[m_idx])

            x = X[:, desc_ids]
            y = X[:, targ_ids]

            self.models[m_idx].fit(x, y)

        return

    def _predict_models(self, X_true):
        X_pred = np.zeros_like(X_true)
        for m_idx in range(self.n_models):
            desc_ids = list(self.desc_ids[m_idx])
            targ_ids = list(self.targ_ids[m_idx])

            y_pred = self.models[m_idx].predict(X_true[:, desc_ids])
            X_pred[:, targ_ids] = y_pred.reshape(-1, len(targ_ids))
        return X_pred

    def _get_residuals(self, X_true):
        X_pred = self._predict_models(X_true)
        return X_true - X_pred

    def _get_scores(self):
        return np.max(self.normalize_residuals(self.residuals),axis=1)

    def _get_grubbs_statistic(self):
        labels = self.labels
        all_residuals = self.residuals
        flt_residuals = all_residuals[labels == 0, :]
        flt_label_idx = np.arange(self.n_instances, dtype=int)[labels == 0]
        nrm_residuals = self.normalize_residuals(flt_residuals)
        degrees_of_freedom = nrm_residuals.shape[0]

        return (
            np.max(nrm_residuals, axis=0),
            flt_label_idx[np.argmax(nrm_residuals, axis=0)],
            degrees_of_freedom,
        )

    @staticmethod
    def _get_grubbs_threshold(dof, critical_t_value):
        factor_01 = (dof - 1) / np.sqrt(dof)
        factor_02 = np.sqrt(
            (critical_t_value ** 2) / (dof - 2 + (critical_t_value ** 2))
        )

        return factor_01 * factor_02

    @staticmethod
    def _get_critical_t_value(a=0.05, dof=100):
        p = 1.0 - a / 2
        return t.ppf(p, dof)

    def _detect_outliers(self):
        grubbs_statistic, potential_outlier_idxs, dof = self._get_grubbs_statistic()
        critical_t_value = self._get_critical_t_value(
            a=self.significance_level, dof=dof - 2
        )
        grubbs_threshold = self._get_grubbs_threshold(dof, critical_t_value)

        if self.verbose:
            msg = """
            potential_outlier_idxs: {}
            critical_t_value(dof={}, significance={}): {}
            grubbs_statistic: {}
            grubbs_threshold: {}
            """.format(
                potential_outlier_idxs,
                dof,
                self.significance_level,
                critical_t_value,
                grubbs_statistic,
                grubbs_threshold,
            )
            print(msg)

        attributes_with_outliers = np.where(grubbs_statistic > grubbs_threshold)[0]
        outlier_idxs = potential_outlier_idxs[attributes_with_outliers]

        return outlier_idxs

    def _update_labels(self, outlier_idxs):
        self.labels[outlier_idxs] = 1
        return

    # Properties
    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        assert isinstance(value, list)
        assert (
            len(value) == self.n_attributes
        ), "The amount of models must equal the amount of attributes"

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
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        assert isinstance(value, np.ndarray)
        assert value.shape[0] == self.n_instances
        self._labels == value
        return

    @property
    def n_anomalies(self):
        return np.sum(self.labels)

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
            assert (
                value <= self.attr_ids
            ), "Nominal attributes have to be a subset of all attributes."
            self._nominal_ids = set(value)
        return

    @property
    def numeric_ids(self):
        return self.attr_ids - self.nominal_ids

