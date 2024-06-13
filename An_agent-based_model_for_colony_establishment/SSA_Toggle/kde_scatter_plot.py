# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Or any other
from scipy.stats import gaussian_kde, binned_statistic, linregress
from scipy.stats import rv_histogram as rv_hist
import sciplot as splt
from typing import Dict

splt.whitegrid()


def kde_plot(xy: np.ndarray, binstep: float = None, axes: plt.axes = None, **kwargs) -> plt.Figure:
    """
    a scatter plot with gaussian estimated kde, binned along x axis and the binned averages are fitted with linear
    regression.

    Parameters
    ----------
    xy : array, [x, y]
        The data for kde plot and bin average

    binstep : float, default 0.2.
        The step length of bin average

    axes: plt.axes
        the axes for plotting

    Returns
    -------
    Fig : plt.Figure
        Return a fig if axes does not given.
    """
    density = gaussian_kde(xy[:1000, :].T)
    color = density(xy.T)
    if axes is None:
        ax = plt.gca()
    else:
        ax = axes
    ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap='coolwarm', **kwargs)
    if binstep is not None:
        bin_num = int(np.ptp(xy[:, 0]) / binstep)
        bin_avg = binned_statistic(xy[:, 0], xy[:, 1], 'mean', bins=bin_num)
        bin_x = np.diff(bin_avg[-2]) / 2 + bin_avg[-2][:-1]
        bin_std = binned_statistic(xy[:, 0], xy[:, 1], 'std', bins=bin_num)
        line_fit = linregress(bin_x, bin_avg[0])
        fit_x = np.linspace(bin_x.min() * 0.8, bin_x.max() * 1.2)
        fit_y = line_fit[0] * fit_x + line_fit[1]
        ax.errorbar(bin_x, bin_avg[0], yerr=bin_std[0],
                    fmt='o', ms=15, fillstyle='none', mew=6, color='#FF8080')

        ax.plot(fit_x, fit_y, c='#5BE3B5',
                label='Slope: %.3f \n $R^{2}$: %.3f' % (line_fit[0], line_fit[2] ** 2))
    ax.grid(False)

    return ax


def hist_sample(sample_data: Dict, bins_num=100, log=True, stat_key=None, xlim=None):
    """
    plot hist
    """
    list_tube_name = list(sample_data.keys())
    axe_num = len(list_tube_name)
    fig_width = 18
    sig_height = fig_width / 4
    fig_height = sig_height * axe_num

    fig, axes = plt.subplots(axe_num, 1, sharex=True, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0)
    data_lim = [.1, -1]
    for index, ax in enumerate(axes):
        if stat_key is not None:
            data = sample_data[list_tube_name[index]][stat_key]
        else:
            data = sample_data[list_tube_name[index]]

        data = data[data > 0]
        data_min, data_max = data.min(), data.max()
        if data_min <= 0:
            data_min = 0.1
        if data_min < data_lim[0]:
            data_lim[0] = data_min
        if data_max > data_lim[-1]:
            data_lim[-1] = data_max

        if log is True:
            bin_range = np.logspace(np.log10(data_lim[0] * .1), np.log10(data_lim[-1] * 1000), bins_num)
            n, bins, patch = ax.hist(data, bins=bin_range, alpha=.5, color='#A6D4FF', histtype='stepfilled')
        else:
            bin_range = np.linspace(*data_lim, num=500)
            n, bins, patch = ax.hist(data, bins=bins_num, alpha=.5, color='#A6D4FF', histtype='stepfilled')

        # hist_of_data = rv_hist((n, bins))
        # ax.plot(bin_range, hist_of_data.pdf(bin_range), color='#8EBAD9')
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(data_lim)
        if log:
            ax.set_xscale('log')
    for index, ax in enumerate(axes):
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        ax.text(x_range[0] * 1.1, y_range[-1] * 0.8, s=f'{list_tube_name[index]}')
    fig.supylabel('Cell count')
    fig.supxlabel('Fluorescent intensity (a.u.)')
    return fig


# %%
if __name__ == '__main__':
    # %%
    data_ps = r'/media/fulab/HAOJIE/noe'
    xlsx_names = [file for file in os.listdir(data_ps) if file.split('.')[-1] == 'xlsx']
    for xlsx in xlsx_names:
        xlsx_ps = os.path.join(data_ps, xlsx)

        data = pd.read_excel(xlsx_ps)
        data_xy = data[['initial length', 'final length']].values

        fig1, ax = plt.subplots(1, 1)
        kde_plot(data_xy, axes=ax)
        ax.set_xlabel('Birth length ($\mu m$)')
        ax.set_ylabel('Division length ($\mu m$)')
        ax.set_title(data_ps.split('\\')[-1])
        ax.legend()
        # fig1.show()
        fig1.savefig(xlsx_ps + '_momo_1.png', transparent=True)

        data_xy = data[['initial length', 'length increment']].values
        fig2, ax2 = plt.subplots(1, 1)
        kde_plot(data_xy, axes=ax2)
        ax2.set_xlabel('Birth length ($\mu m$)')
        ax2.set_ylabel('Length increment ($\mu m$)')
        ax2.set_title(data_ps.split('\\')[-1][0:-5])
        ax2.legend()
        # fig2.show()
        fig2.savefig(xlsx_ps + '_momo_2.png', transparent=True)
