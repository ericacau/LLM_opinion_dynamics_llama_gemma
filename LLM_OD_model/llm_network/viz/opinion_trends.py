import os
import matplotlib as mpl
import json
from collections import defaultdict, Counter
import numpy as np
# from .plot_config import apply_plot_config


if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")
import matplotlib.pyplot as plt

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"

class OpinionTrends(object):
    def __init__(self, filename: object):
        self.data = defaultdict(lambda: defaultdict(int))
        self.x = []
        self.y = defaultdict(list)
        old_iter = -1
        statuses = {}
        
        try:
            with open(filename) as file:                
                for id_row, l in enumerate(file):
                    try:
                        l = json.loads(l)
                        iter = l["iteration"]
                        
                        # Store statuses for current iteration
                        if "status" in l:
                            for agent, status in l["status"].items():
                                statuses[agent] = status
                                
                        # Process data when iteration changes
                        if iter != old_iter and old_iter != -1:
                            # Calculate status distribution
                            status_counts = Counter(statuses.values())
                            total_agents = len(statuses)
                            
                            # Store percentages
                            self.x.append(old_iter)
                            for opinion in range(7):  # 0 to 6
                                percentage = (status_counts[opinion] / total_agents) * 100
                                self.y[opinion].append(percentage)
                            
                            statuses = {}  # Reset for next iteration
                            
                        old_iter = iter
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON at line {id_row}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line {id_row}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading file: {e}")
            
    def plot(self, ax=None, limit=None):
        if ax is None:
            ax = plt.gca()
            
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        labels = ['Strongly disagree', 'Disagree', 'Mildly disagree', 'Neutral', 
                  'Mildly agree', 'Agree', 'Strongly agree']
        
        if limit:
            x = self.x[:limit]
        else:
            x = self.x
            
        lines = []
        for opinion in range(7):
            if limit:
                y = self.y[opinion][:limit]
            else:
                y = self.y[opinion]
                
            if len(x) > 0 and len(y) > 0:  # Only plot if we have data
                line, = ax.plot(x, y, color=colors[opinion], label=labels[opinion])
                lines.append(line)
                
        ax.set_xlabel("Iterations")
        ax.set_ylabel("% Agents")
        ax.grid(True)
        
        return lines
