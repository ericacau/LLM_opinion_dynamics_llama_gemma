import os
import matplotlib as mpl
import json
from collections import defaultdict
import numpy as np

if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class OpinionVolume(object):
    def __init__(self, filename: object):
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """

        # iter -> agent -> opinion
        self.data = defaultdict(lambda: defaultdict(float))
        self.agent_type =  defaultdict(str)

        with open(filename) as file:
            old_iter = 0
            statuses = {}

            print(filename)

            for id_row, l in enumerate(file):
                try:
                    l = json.loads(l)
                    iter = l["iteration"]

                    agent = l['interacting_agents']['discussant']
                except:
                    continue

                iter = l["iteration"]
                agent = l['interacting_agents']['discussant']
                opinion = int(l['status'][agent])

                self.data[iter][agent] = opinion
                self.agent_type[agent] = agent = l['interacting_agents']['discussant_llm']

        models = set(list(self.agent_type.values()))

        # class -> iter -> mean value
        res = defaultdict(lambda: defaultdict(list))
        for i in self.data:
            for u in self.data[i]:
                ag_type = self.agent_type[u]
                res[ag_type][i].append(self.data[i][u])

        
        # normalize
        self.fres = defaultdict(lambda: defaultdict(float))
        self.sfres = defaultdict(lambda: defaultdict(float))
        for at in res:
            for it in res[at]:
                self.fres[at][it] = np.mean(res[at][it])
                self.sfres[at][it] = np.std(res[at][it])


       

    def plot(self, filename=None, limit=None, flag=None):
        """
        Generates the plot

        :param filename: Output filename
        :param percentile: The percentile for the trend variance area
        """

        if flag is not None:
            flag = flag.split(",")

        for ag, values in self.fres.items():
            if ag == flag[0]:
                x , y, e = [0], [0], [0]
            else:
                x , y, e = [0], [6], [0]

            x.extend(list(values.keys())[:limit])
            y.extend(list(values.values())[:limit])
            e.extend(list(self.sfres[ag].values())[:limit])

            plt.errorbar(x, y, e, label=ag, alpha=0.9)

        plt.xlabel("Iterations", fontsize=10)
        plt.ylabel("LLM-agents mean opinion", fontsize=10)
        plt.legend(loc="best", fontsize=8, ncol=4, bbox_to_anchor=(0.9, 1.15))

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()
