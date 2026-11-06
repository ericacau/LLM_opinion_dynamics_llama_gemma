import os
import matplotlib as mpl
import json
from collections import defaultdict
import numpy as np
from .plot_config import  apply_plot_config

#if os.environ.get("DISPLAY", "") == "":
#    print("no display found. Using non-interactive Agg backend")
#    mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class OpinionArea(object):
    def __init__(self, filename: object):
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """

        self.data = defaultdict(list)

        with open(filename) as file:
            old_iter = 0
            statuses = {}

            print(filename)

            for id_row, l in enumerate(file):
                try:
                    l = json.loads(l)
                    # try:

                    iter = l["iteration"]
                except:
                    continue

                if iter == old_iter:
                    statuses = l["status"]
                else:
                    sts = {
                        k: sum(value == k for value in statuses.values())
                        for k in range(0, 7)
                    }
                    old_iter = iter
                    for k, v in sts.items():
                        self.data[k].append(v)

        for k, v in self.data.items():
            self.data[k] = [x / len(l["status"]) for x in self.data[k]]

    def plot(self, filename=None, limit=None, legend=False, ax=None):
        """
        Generates the plot

        :param filename: Output filename
        :param percentile: The percentile for the trend variance area
        """

        label = {
            0: "Strongly Disagree",
            1: "Disagree",
            2: "Mildly Disagree",
            3: "Neutral",
            4: "Mildly Agree",
            5: "Agree",
            6: "Strongly Agree",
        }
        categories = [0, 1, 2, 3, 4, 5, 6]
        data = [self.data[k][:limit] if limit else self.data[k] for k in categories]
        x = range(0, len(data[0]))

        # Map numbers to a diverging color palette (red to green)
        colors = plt.get_cmap('RdYlGn')(np.linspace(0, 1, len(categories)))

        owns_axis = ax is None
        if ax is None:
            fig, ax = plt.subplots()

        ax.stackplot(x, data, labels=[label[k] for k in categories], colors=colors, alpha=0.7)
        ax.set_xlabel("Iterations", fontsize=14)
        ax.set_ylabel("% Agents", fontsize=14)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)

        if legend:
            ax.legend(loc="best", fontsize=9, ncol=4, bbox_to_anchor=(1.01, 1.18))

        if filename is not None and owns_axis:
            apply_plot_config()
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            plt.savefig(filename)
            plt.close(fig)
        elif owns_axis:
            apply_plot_config()
            plt.show()
