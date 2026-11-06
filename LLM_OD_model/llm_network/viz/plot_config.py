from matplotlib import rcParams
import matplotlib.pyplot as plt

def apply_plot_config():
    plt.style.use('ggplot')

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.style'] = 'normal'

    rcParams['figure.facecolor'] = 'white'
    rcParams['figure.dpi'] = 300

    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.transparent'] = True

    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.labelsize'] = 13
    rcParams['axes.labelcolor'] = 'black'
    rcParams['axes.edgecolor'] = 'black'
    rcParams['axes.linewidth'] = 1
    rcParams['axes.facecolor'] = 'white'

    rcParams['legend.fontsize'] = 11

    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    rcParams['xtick.major.size'] = 2
    rcParams['ytick.major.size'] = 2
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11

    rcParams['lines.linewidth'] = 1
    rcParams['lines.markersize'] = 5

    rcParams['grid.color'] = 'white'
    rcParams['grid.linewidth'] = 0.0
