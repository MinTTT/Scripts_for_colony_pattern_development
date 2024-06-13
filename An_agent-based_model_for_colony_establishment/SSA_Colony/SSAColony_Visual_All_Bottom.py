# -*- coding: utf-8 -*-

"""
 Plot the colony bottom for a cell file.

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""
# %%
# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
# […]
# Own module

from src.toggle import pyrunSim, pyrunMultSim, pyrunBatchSim
from CellMD3D_visulization.CellMD3D_bottom_visual import cell_vertices, cell_path
import matplotlib.patches as mpatches

from SSAColony_R2G_BatchRun import read_cells_rets

# %%
RedColor = np.array((246, 61, 41)) / 255
GreenColor = np.array((41, 246, 61)) / 255
growth_rate = 1.6
size = 100000
green = np.ones(size, dtype=int) * 20
red = np.ones(size, dtype=int) * 50
time_length = np.log(2) / growth_rate * 10
time_step = .1
plot_flag = True
test_ret = pyrunMultSim(growth_rate, green, red, time_length, time_step, threadNum=22)
ratio_stat = (test_ret[:, 1, :] + .1) / (test_ret[:, 2, :] + .1)  # (Green + 1) / (Red +1 )
ratio_stat = np.log(ratio_stat)
gmm = GaussianMixture(n_components=2, random_state=0, verbose=1)
# gmm.fit(ratio_stat[-1, :].reshape(-1, 1))
gmm.fit(ratio_stat[-1, :].reshape(-1, 1))

class_mean = gmm.means_
green_label = 0 if class_mean[0] > class_mean[1] else 1
test_cls = gmm.predict(ratio_stat[-2, :].reshape(-1, 1))

# %%

file_ps = r'Y:\Data_Raid\sunhui_code_ret\ssa_in26_colony_RunBatch.7.46\Cells\22.txt'

cells = read_cells_rets(file_ps)

cells_all_location = np.array([cell.center for cell in cells])  # [cells, loactionXYZ]
cells_all_R = np.array([cell.R for cell in cells])
cells_all_G = np.array([cell.G for cell in cells])
cells_all_lambda = np.array([cell.GrowthRate for cell in cells])
cells_all_states = gmm.predict(np.log((cells_all_G + 1.) / (cells_all_R + 1.)).reshape(-1, 1))
# %
# ============== Parameters for Draw =============================#
z_top = 4
z_bottom = -4
location_mask = np.logical_and(cells_all_location[..., -1] > z_bottom, cells_all_location[..., -1] < z_top)
location_index = np.where(location_mask == True)
range_factor = 1.1
x_bin_length = 5
y_bin_length = 5
z_bin_length = 2

cells_p = [cell_path(0.7, cells[cell_i].p[:-1], cells[cell_i].q[:-1]) for cell_i in location_index[0]]

cells_location = cells_all_location[location_mask, ...]
cells_R = cells_all_R[location_mask]
cells_G = cells_all_G[location_mask]
cells_states = cells_all_states[location_mask]
locations_min = cells_location.min(axis=0)
locations_max = cells_location.max(axis=0)
colony_center = np.median(cells_location, axis=0)

fig1_colony_bottom, ax1 = plt.subplots(1, 1, figsize=(15, 15))
red_cells_loc = cells_location[cells_states != green_label]
green_cells_loc = cells_location[cells_states == green_label]
for cell_i, cell_p in enumerate(cells_p):
    if cells_states[cell_i] == green_label:
        cell_color = tuple(GreenColor)
    else:
        cell_color = tuple(RedColor)
    ax1.add_patch(mpatches.PathPatch(cell_p, facecolor=cell_color, edgecolor='k', alpha=0.6))
# ax1.scatter(green_cells_loc[:, 0], green_cells_loc[:, 1], color=tuple(GreenColor), s=65, alpha=.6)
# ax1.scatter(red_cells_loc[:, 0], red_cells_loc[:, 1], color=tuple(RedColor), s=65, alpha=.6)
ax1.set_xlim(-150, 150)
ax1.set_ylim(-150, 150)

fig1_colony_bottom.show()

graphPad_data = np.empty((len(cells_location), 3)) * np.nan
graphPad_data[:len(green_cells_loc), 0] = green_cells_loc[:, 0]
graphPad_data[len(green_cells_loc):, 0] = red_cells_loc[:, 0]
graphPad_data[:len(green_cells_loc), 1] = green_cells_loc[:, 1]
graphPad_data[len(green_cells_loc):, 2] = red_cells_loc[:, 1]
