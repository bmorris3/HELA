import os

import numpy as np
from sklearn import metrics, multioutput
import joblib

from .dataset import load_dataset, load_data_file
from .models import Model
from .plot import predicted_vs_real, feature_importances, posterior_matrix

__all__ = ['RandomForest']


def train_model(dataset, num_trees, num_jobs, verbose=1):
    pipeline = Model(num_trees, num_jobs,
                     names=dataset.names,
                     ranges=dataset.ranges,
                     colors=dataset.colors,
                     verbose=verbose)
    pipeline.fit(dataset.training_x, dataset.training_y)
    return pipeline


def test_model(model, dataset, output_path):
    if dataset.testing_x is None:
        return

    pred = model.predict(dataset.testing_x)
    r2scores = {name_i: metrics.r2_score(real_i, pred_i)
                for name_i, real_i, pred_i in
                zip(dataset.names, dataset.testing_y.T, pred.T)}
    print("Testing scores:")
    for name, values in r2scores.items():
        print("\tR^2 score for {}: {:.3f}".format(name, values))

    fig = predicted_vs_real(dataset.testing_y, pred, dataset.names,
                            dataset.ranges)
    fig.savefig(os.path.join(output_path, "predicted_vs_real.pdf"),
                bbox_inches='tight')
    return fig


def compute_feature_importance(model, dataset, output_path):
    regr = multioutput.MultiOutputRegressor(model, n_jobs=1)
    regr.fit(dataset.training_x, dataset.training_y)

    forests = [i.rf for i in regr.estimators_] + [model.rf]
    # for i, forest_i in enumerate(forests):
    #     np.save('feature_importance_{0}.npy'.format(i),
    #             forest_i.feature_importances_)
    #     # TODO: feature importances of shape (N, )
    #     # where N is the number of "features"

    fig = feature_importances(
                forests=[i.rf for i in regr.estimators_] + [model.rf],
                names=dataset.names + ["joint prediction"],
                colors=dataset.colors + ["C0"])

    fig.savefig(os.path.join(output_path, "feature_importances.pdf"),
                bbox_inches='tight')


def prediction_ranges(preds):
    percentiles = (np.percentile(pred_i, [50, 16, 84]) for pred_i in preds.T)
    return np.array([(a, c - a, a - b) for a, b, c in percentiles])


class RandomForest(object):
    """
    A class for a random forest.
    """
    def __init__(self, training_dataset, model_path, data_file):
        """
        Parameters
        ----------
        training_dataset
        model_path
        data_file
        """
        self.training_dataset = training_dataset
        self.model_path = model_path
        self.data_file = data_file
        self.output_path = self.model_path

        self.dataset = None
        self.model = None

    def train(self, num_trees=1000, num_jobs=5, quiet=False):
        """
        Train the random forest on a set of observations.

        Parameters
        ----------
        num_trees
        num_jobs
        quiet
        kwargs

        Returns
        -------
        fig :
        """
        # Loading dataset
        self.dataset = load_dataset(self.training_dataset)

        # Training model
        self.model = train_model(self.dataset, num_trees, num_jobs, not quiet)

        os.makedirs(self.model_path, exist_ok=True)
        model_file = os.path.join(self.model_path, "model.pkl")
        # Saving model
        joblib.dump(self.model, model_file)

        # Printing model information...
        print("OOB score: {:.4f}".format(self.model.rf.oob_score_))

        fig = test_model(self.model, self.dataset, self.model_path)

        return fig

    def feature_importance(self, model, dataset):
        return compute_feature_importance(model, dataset, self.model_path)

    def predict(self, plot_posterior=True):
        """
        Predict values from the trained random forest.

        Parameters
        ----------
        plot_posterior

        Returns
        -------
        preds : `~numpy.ndarray`
            N x M values where N is number of parameters, M is number of
            samples/trees (check out attributes of model for metadata)
        """
        model_file = os.path.join(self.model_path, "model.pkl")
        # Loading random forest from '{}'...".format(model_file)
        model = joblib.load(model_file)

        # Loading data from '{}'...".format(data_file)
        data, _ = load_data_file(self.data_file, model.rf.n_features_)

        # TODO: This is the line to save, it's an array of
        #
        preds = model.trees_predict(data[0])
        # np.save('preds.npy', preds)

        pred_ranges = prediction_ranges(preds)

        for name_i, pred_range_i in zip(model.names, pred_ranges):
            print("Prediction for {}: {:.3g} "
                  "[+{:.3g} -{:.3g}]".format(name_i, *pred_range_i))

        if plot_posterior:
            # Plotting and saving the posterior matrix..."
            fig = posterior_matrix(preds, None,
                                   names=model.names,
                                   ranges=model.ranges,
                                   colors=model.colors)
            os.makedirs(self.output_path, exist_ok=True)
            fig.savefig(os.path.join(self.output_path, "posterior_matrix.pdf"),
                        bbox_inches='tight')
        return preds
