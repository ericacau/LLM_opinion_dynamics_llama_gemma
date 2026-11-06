import os
import matplotlib as mpl
import json
from collections import defaultdict
import numpy as np
from .plot_config import apply_plot_config
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib import cm


if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"



class StackedAreaChart: 
    def __init__(self, filename: object):
        
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """        
        self.data = defaultdict(list)
        try: 
            with open(filename) as file:
                old_iter = 0
                statuses = {}

                for id_row, l in enumerate(file):
                    try:
                        l = json.loads(l)
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
        except:
            print(filename)    

        for k, v in self.data.items():
            self.data[k] = [x / len(l["status"]) for x in self.data[k]]


    def plot(self, ax, filename=None, limit=None):
        """
        Generates a stacked area chart.
        :param ax: Matplotlib axis object
        :param filename: Output filename (optional)
        :param limit: Number of iterations to show (optional)
        """
        cmap = get_cmap("RdYlGn")  
        norm = Normalize(vmin=0, vmax=6)
        colors = {k: mpl.colors.to_hex(cmap(norm(k))) for k in range(7)}
        label = {
            0: "Strongly Disagree",
            1: "Disagree",
            2: "Mildly Disagree",
            3: "Neutral",
            4: "Mildly Agree",
            5: "Agree",
            6: "Strongly Agree",
        }
        
        # Order for stacking
        order = [0, 1, 2, 3, 4, 5, 6]
        x = range(0, len(self.data[0])) if limit is None else range(0, limit)
        y = [self.data[k][:limit] if limit else self.data[k] for k in order]
        labels = [label[k] for k in order]
        color_list = [colors[k] for k in order]
        
        # NORMALIZE DATA: Ensure y values sum to 1 at each x point
        import numpy as np
        y_array = np.array(y)
        y_sums = np.sum(y_array, axis=0)
        # Avoid division by zero
        y_sums[y_sums == 0] = 1
        y_normalized = y_array / y_sums
        
        # Create the stackplot with normalized data
        ax.stackplot(x, y_normalized, labels=labels, colors=color_list, alpha=0.6)
        
        # FIX: Set exact limits to remove ALL white space
        if limit is None:
            ax.set_xlim(0, len(self.data[0]) - 1)
        else:
            ax.set_xlim(0, limit - 1)
        
        # Set exact y limits with no white space
        ax.set_ylim(0, 1)  # Exact bounds, no padding
        
        ax.set_ylabel("")
        ax.legend().set_visible(False)
        
        # Remove ALL margins and padding
        ax.margins(0)
        
        return ax.get_figure()

    # Alternative version with more control:
    def plot_no_whitespace(self, ax, filename=None, limit=None):
        """
        Generates a stacked area chart with no white space at edges.
        """
        cmap = get_cmap("RdYlGn")  
        norm = Normalize(vmin=0, vmax=6)
        colors = {k: mpl.colors.to_hex(cmap(norm(k))) for k in range(7)}
        label = {
            0: "Strongly Disagree",
            1: "Disagree", 
            2: "Mildly Disagree",
            3: "Neutral",
            4: "Mildly Agree",
            5: "Agree",
            6: "Strongly Agree",
        }
        
        order = [0, 1, 2, 3, 4, 5, 6]
        
        # Determine data length
        data_length = len(self.data[0]) if limit is None else limit
        
        # Create x values as numpy array for better control
        import numpy as np
        x = np.arange(0, data_length)
        y = [self.data[k][:limit] if limit else self.data[k][:data_length] for k in order]
        labels = [label[k] for k in order]
        color_list = [colors[k] for k in order]
        
        # Create stackplot
        ax.stackplot(x, y, labels=labels, colors=color_list, alpha=0.6)
        
        # Set exact limits with no padding
        ax.set_xlim(0, data_length - 1)
        ax.set_ylim(0, 1)  # Changed from -0.01, 1.01 to exact 0, 1
        ax.margins(0)  # Remove all margins
        
        ax.set_ylabel("")
        ax.legend(loc='upper left')
        
        return ax.get_figure()

    # If you want to be very precise about eliminating ALL white space:
    def plot_tight(self, ax, filename=None, limit=None):
        """
        Generates a stacked area chart with absolutely no white space.
        """
        cmap = get_cmap("RdYlGn")  
        norm = Normalize(vmin=0, vmax=6)
        colors = {k: mpl.colors.to_hex(cmap(norm(k))) for k in range(7)}
        label = {
            0: "Strongly Disagree",
            1: "Disagree",
            2: "Mildly Disagree", 
            3: "Neutral",
            4: "Mildly Agree",
            5: "Agree",
            6: "Strongly Agree",
        }
        
        order = [0, 1, 2, 3, 4, 5, 6]
        data_length = len(self.data[0]) if limit is None else limit
        
        x = range(0, data_length)
        y = [self.data[k][:limit] if limit else self.data[k][:data_length] for k in order]
        labels = [label[k] for k in order]
        color_list = [colors[k] for k in order]
        
        ax.stackplot(x, y, labels=labels, colors=color_list, alpha=0.6)
        
        # Tight limits - no white space
        ax.set_xlim(0, data_length - 1)
        ax.set_ylim(0, 1)
        ax.margins(0)
        
        # Additional settings to ensure no white space
        ax.autoscale(tight=True)
        ax.set_aspect('auto')
        
        ax.set_ylabel("")
        ax.legend(loc='upper left')
        
        return ax.get_figure()