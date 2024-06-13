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
import pandas as pd
import numpy as np  # Or any other
from src.toggle import pyrunSim, pyrunMultSim, pyrunBatchSim
from tqdm import tqdm
import matplotlib.pyplot as plt
# […]
from joblib import Parallel, delayed
# Own modules
import sciplot as splt
from kde_scatter_plot import kde_plot
from scipy.stats import binned_statistic, gamma
from sklearn.mixture import GaussianMixture
from joblib import dump, load
from tqdm import tqdm
splt.whitegrid()

import datetime
def RG2Predict(green_signal, red_signal):
    return np.log((green_signal + 2) / (red_signal + 2))


RedColor = np.array((241, 148, 138)) / 255
GreenColor = np.array((130, 224, 170)) / 255
# %%

results_directory = r'./Data/single_cell_transition_rate'
if not os.path.isdir(results_directory):
    os.makedirs(results_directory)

size = 100000
green = np.ones(size, dtype=int) * 20
red = np.ones(size, dtype=int) * 50
growth_rate = .18
time_length = 400  # np.log(2) / growth_rate * 10
time_step = .1
test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=60)

ratio_stat = RG2Predict(test_ret[:, 1, :], test_ret[:, 2, :])
gmm = GaussianMixture(n_components=2, random_state=0, verbose=1)
# gmm.fit(ratio_stat[-1, :].reshape(-1, 1))
gmm.fit(ratio_stat[-1, :].reshape(-1, 1))

class_mean = gmm.means_

green_label = 0 if class_mean[0] > class_mean[1] else 1

test_cls = gmm.predict(ratio_stat[-2, :].reshape(-1, 1))

# ======== show the predict rets =============== #
fig3, ax3 = plt.subplots(1, 1)
green_cells_mask = test_cls == green_label

ax3.scatter(test_ret[-2, 2][green_cells_mask], test_ret[-2, 1][green_cells_mask],
            color=GreenColor, alpha=.1)
ax3.scatter(test_ret[-2, 2][~green_cells_mask], test_ret[-2, 1][~green_cells_mask],
            color=RedColor, alpha=.1)
ax3.set_xlim(1, 500)
ax3.set_ylim(1, 500)
# ax3.set_xscale('log')
# ax3.set_yscale('log')
ax3.set_xlabel('Red')
ax3.set_ylabel('Green')
splt.aspect_ratio(1)
fig3.show()

# %%

red_init = np.linspace(10, 80, num=8).astype(int)
green_init = np.linspace(1, 10, num=8).astype(int)
vx, vy = np.meshgrid(red_init, green_init)
# parameter_dict = [[15, 55],  # green, red
#                   [12, 55],
#                   [9, 55],
#                   [6, 55],
#                   [3, 55],
#                   [1, 55]]
parameter_dict = [[green, red] for green, red in zip(vy.flatten(),
                                                     vx.flatten())]
results_dict = []

transTimeList = []
for par in tqdm(parameter_dict):
    size = 50000
    green = np.ones(size, dtype=int) * par[0]
    red = np.ones(size, dtype=int) * par[1]
    growth_rate = 1.6
    time_length = np.log(2) / growth_rate * 5000
    time_step = .1  # .1
    print(f"Start Sim: {par}. \n")
    ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=60)
    # Attention! Don't save simulation raw data, it's too big!
    # print(f"End Sim: {par}. \n")
    # time_now = datetime.datetime.now().strftime('%Y-%M-%d-%H-%m-%S')
    # dump(ret, os.path.join(results_directory, f'cells_stats_G{par[0]}_R{par[1]}_lambda{growth_rate}_{time_now}.pkl'))
    ratio_stat = RG2Predict(ret[:, 1, :], ret[:, 2, :])
    class_pred = gmm.predict(ratio_stat.reshape(-1, 1))
    green_mask = class_pred == green_label
    green_mask = green_mask.reshape(ratio_stat.shape)  # [Time, Cell]
    green_number = np.sum(green_mask, axis=1)
    delta_green_ratio = np.diff(green_number) / (size - green_number[:-1]) / time_step

    time_list = np.arange(len(green_number)) * time_step
    time_diff = time_list[1:]
    green_ratio = green_number / size

    # state_state = class_pred.reshape(ratio_stat.shape)  # cell states?
    stateChange = np.vstack([np.zeros((1, green_mask.shape[1])), green_mask])
    stateChange = np.diff(stateChange, axis=0)
    # stateTransTimeIndex, stateTransCellIndex = np.where(np.diff(stateChange, axis=0) == 1)
    TransTime = np.zeros(size) * np.nan
    for i in range(size):
        TransTime[i] = np.argmax(stateChange[:, i] == 1) * time_step  # tau R->G

    transTimeList.append(TransTime.reshape(-1, 1))

    graphPad_data = np.zeros((len(time_list) + len(time_diff), 3)) * np.nan
    graphPad_data[:len(time_list), 0] = time_list
    graphPad_data[:len(time_list), 1] = green_ratio
    graphPad_data[len(time_list):, 0] = time_diff
    graphPad_data[len(time_list):, 2] = delta_green_ratio
    results_dict.append(graphPad_data.copy())


# %% statistic of tau (first passage time)
tuaDistributionList = []
meanTau = []
meanTauEstimate = []
sampleNameList = []
gammaAlpha = []
gammaScale = []
for tuaIndex, tua in enumerate(transTimeList):
    nanmask = ~np.isnan(tua)
    nanNum = np.sum(nanmask == False)
    bincount, edge, _ = binned_statistic(tua[nanmask], tua[nanmask], 'count', bins=50, range=(0, int(time_length) - 1))
    midPoint = (edge[1:] - edge[:-1]) / 2 + edge[:-1]
    gridWidth = edge[1] - edge[0]
    # tuaDistribution = np.hstack([midPoint.reshape(-1, 1), (bincount / size).reshape(-1, 1)])
    tuaDistributionList.append((bincount / size / gridWidth).reshape(-1, 1))
    tauMean = tua[nanmask].mean()
    meanTau.append(tua[nanmask].mean())
    print(f"Mean tau: {'%.2f' % tauMean}")
    parameter = parameter_dict[tuaIndex]
    sampleNameList.append(f"#{tuaIndex + 1} ({parameter[1]}, {parameter[0]}) <tau> {'%.2f' % tauMean}")
    distPars = gamma.fit(tua[nanmask], floc=0)
    meanEstimate = gamma.mean(*distPars)
    gammaAlpha.append(distPars[0])
    gammaScale.append(distPars[2])
    meanTauEstimate.append(meanEstimate)
    print(distPars, meanEstimate)

tuaDistributionList = np.hstack([midPoint.reshape(-1, 1)] + tuaDistributionList)

meanTau = np.array(meanTau).reshape(vx.shape)
meanTau = meanTau[::-1, ...]
meanTauEstimate = np.array(meanTauEstimate).reshape(vx.shape)
meanTauEstimate = meanTauEstimate[::-1, ...]
tuaDistributionDataFrame = pd.DataFrame(data=tuaDistributionList, columns=['Time'] + sampleNameList)

# %% plot tau pdf and estimated parameters

fig2, ax2 = plt.subplots(1, 1)
index = 8 * 1 + 0
tau = transTimeList[index]
initPar = parameter_dict[index]
ax2.hist(tau[~np.isnan(tau)], bins=500, density=True, histtype='stepfilled', alpha=0.2, log=True)
ax2.plot(np.linspace(0, 1000, num=1000),
         gamma.pdf(np.linspace(0, 1000, num=1000), a=gammaAlpha[index], scale=gammaScale[index]), '--',
         label=f'Init. ({initPar[-1]}, {initPar[0]})')
index = 8 * 0 + 7
initPar = parameter_dict[index]
tau = transTimeList[index]
ax2.hist(tau[~np.isnan(tau)], bins=500, density=True, histtype='stepfilled', alpha=0.2, log=True)
ax2.plot(np.linspace(0, 1000, num=1000),
         gamma.pdf(np.linspace(0, 1000, num=1000), a=gammaAlpha[index], scale=gammaScale[index]), '--',
         label=f'Init. ({initPar[-1]}, {initPar[0]})')
ax2.set_xlim(1, 500)
ax2.set_xscale('log')
ax2.set_yscale('linear')
ax2.set_ylim(0, 0.2)
plt.locator_params(axis='y', nbins=4)
ax2.legend()
fig2.show()

# %% Plot tau with cell states

fig3, ax3 = plt.subplots(1, 1, figsize=(12, 12))
# ax3.scatter(test_ret[-2, 2], test_ret[-2, 1], alpha=.1, s=10)
kde_plot(np.vstack([test_ret[-2, 2], test_ret[-2, 1]]).T[::10])
ax3.set_xlim(1, 300)
ax3.set_ylim(1, 300)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Red')
ax3.set_ylabel('Green')
ax3.scatter(vx, vy, s=200, color='#8CF18A', edgecolors='k', linewidth=5, alpha=.5)

splt.aspect_ratio(1)
fig3.show()
