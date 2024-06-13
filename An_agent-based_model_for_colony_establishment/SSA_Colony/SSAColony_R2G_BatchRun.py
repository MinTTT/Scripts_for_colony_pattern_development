"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys
sys.path.insert(0, r'./')
# […]
import time

from tqdm import tqdm
# Libs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Or any other


from matplotlib.colors import LinearSegmentedColormap

import sciplot as splt
import threading
from src.toggle import pyrunSim, pyrunMultSim, pyrunBatchSim
from sklearn.mixture import GaussianMixture
from typing import TextIO

from subprocess import PIPE, Popen

red = np.array([231, 76, 60]) / 255
white = np.array([1, 1, 1])
green = np.array([93, 173, 226]) / 255
nodes = [0.0, .5, 1.0]
RedColor = np.array((241, 148, 138)) / 255
GreenColor = np.array((130, 224, 170)) / 255
RedGreen_cmap = LinearSegmentedColormap.from_list('RedGreen', list(zip(nodes, [red, white, green])))

splt.whitegrid()
# […]

# Own modules

# split cells file
paras_location = [0, 1, 2, slice(3, 6), slice(6, 9), 9, slice(10, 13), slice(13, 16), 16, slice(17, 20),
                  slice(20, 23),
                  23, 24, 25, 26, 27, 28, 29]


# |t 0|ID|Type|p|q|Length|Tensor|Velocity|GrowthRate|DynFric|StaFric|time_p|time_q|age_p|age_q|Ancestor|G|R|

def read_cells_rets(CellFilePath, pbar=True):

    lines = []
    if pbar:
        p_bar = tqdm()
    read_file_flag = True
    reopen_flag = False
    while read_file_flag:
        with open(CellFilePath) as ret_file:
            while True:
                line = ret_file.readline()
                if line == '':
                    break
                try:
                    cell_pars = [float(par) for par in line.replace('\n', '').split(' ')]
                    cell_parameters = [np.array(cell_pars[index]) if isinstance(index, slice) else cell_pars[index]
                                       for index in paras_location]
                except IndexError:
                    print('File is writing now, reading again')
                    # re-open file
                    reopen_flag = True
                    break
                lines.append(Cell(*cell_parameters))
                if pbar:
                    p_bar.update()
        # check file size reopen again
        time.sleep(1)
        with open(CellFilePath) as ret_file:
            line_number = 0
            while True:
                line = ret_file.readline()
                if line == '':
                    break
                line_number += 1
        if line_number == len(lines) and line_number >= 1:
            read_file_flag = False

    return lines


class Cell:

    def __init__(self, t, ID, Type,  # 0, 1, 2
                 p: np.array,  # 3:6
                 q: np.array,  # 6:9
                 Length,  # 9
                 T: np.array,  # 10:13
                 Velocity: np.array,  # 13:16
                 GrowthRate,  # 16
                 DynFric, StaFric,  # slice(17, 20), slice(20, 23),
                 time_p, time_q,
                 age_p, age_q,
                 Ancestor, G, R):
        self.t = t
        self.ID = ID
        self.Type = Type
        self.p = p
        self.q = q
        self.Length = Length
        self.T = T
        self.Velocity = Velocity
        self.GrowthRate = GrowthRate
        self.DynFric = DynFric
        self.StaFric = StaFric
        self.time_p = time_p
        self.time_q = time_q
        self.age_p = age_p
        self.age_q = age_q
        self.Ancestor = Ancestor
        self.G = G
        self.R = R
        self.state = None

        self.center = (self.p + self.q) / 2


def loadAllCells(CellFilesPath):
    """Load all cells data from directory.

    Parameters:
        CellFilesPath (str): the PATH for cells data.

    Returns:
        data (List[List[Cell]]): list(list(Cell obj)
    """

    def readCellTarget(CellFilePath, buff, index):
        buff[index] = read_cells_rets(CellFilePath, False)

        return None

    filesDir = [file.name for file in os.scandir(CellFilesPath) if file.is_file()]
    file_number = len(filesDir)

    data = [None] * file_number
    read_threads = [threading.Thread(target=readCellTarget, args=(os.path.join(CellFilesPath, path), data, i)) for
                    i, path in enumerate(filesDir)]
    for read_thread in read_threads:
        read_thread.start()

    pbar = tqdm(total=len(filesDir))
    start_count = 0
    while True:
        count = 0
        for i in data:
            if i is not None:
                count += 1
        if start_count < count:
            pbar.update(count - start_count)
            start_count = count
        if start_count == file_number:
            break

    return data


def value2color(data, color):
    color_value = np.zeros(data.shape + (3,))
    rgb_color = np.array(color) / 255
    norm_data = data / np.ptp(data)
    for i in range(3):
        color_value[..., i] = norm_data * rgb_color[i]
    return color_value


def value2color_cmap(data, cmap):
    # color_value = np.zeros(data.shape + (3,))
    # data_mask = ~np.isnan(data)
    # data_norm = (data - data[data_mask].min()) / np.ptp(data[data_mask])
    normalized_data = data_norm(data)
    color_value = cmap(normalized_data)
    return color_value


def data_norm(data):
    data_mask = ~np.isnan(data)
    data_norm = (data - data[data_mask].min()) / np.ptp(data[data_mask])
    return data_norm


def cells2cc1(filePath, save_dir=None):
    """Convert the Cell data to cc1 file for rendering in PyMol.

    .cc1 file format:
    row1: atom total number;
    col1: atom type; col2: atom index; col3: x, col4: y; col5: z; col6 force; col7 bond.

    Parameters
    --------
    filePath: string
        the exported simulation data.

    save_dir: string or None, default None
        the directory specified for saving the .cc1 file.

    """
    cells = read_cells_rets(filePath)
    fileName = os.path.basename(filePath)
    fileName = '.'.join([fileName.split('.')[0], 'cc1'])
    total_atoms = 2 * len(cells)

    if save_dir is None:
        save_dir = os.path.dirname(filePath)
    else:
        try:
            os.mkdir(save_dir)
            print(f"Make dir: {save_dir}")
        except FileExistsError:
            pass
    with open(os.path.join(save_dir, fileName), 'w') as file:
        file.write(f'{total_atoms}\n')
        for i, cell in enumerate(cells):
            str1 = f"C {i * 2 + 1} {' '.join(cell.p.astype(str))} 1 {i * 2 + 2}\n"
            str2 = f"C {i * 2 + 2} {' '.join(cell.q.astype(str))} 1 {i * 2 + 1}\n"
            file.write(str1)
            file.write(str2)
    return None


def cartesian2polar(yv, xv, center):
    """
    the center should have axis0, axis1 order (y, x).

    Args:
        yv (np.ndarray): y indexes;
        xv (np.ndarray): x indexes;
        center (array-like): (y, x);

    Returns:
        tuple: pho, phi
    """
    dyv = yv - center[0]
    dxv = xv - center[1]
    pho = np.sqrt(dyv ** 2 + dxv ** 2)
    phi = np.arccos(dyv / pho)
    phi[dxv < 0] = 2 * np.pi - phi[dxv < 0]
    return pho, phi


def binned_along_radius(pho, data, bins=np.linspace(0, 1100, num=100)):
    """_summary_

    Args:
        pho (_type_): _description_
        data (_type_): _description_
        bins (_type_, optional): _description_. Defaults to np.linspace(0, 1100, num=100).

    Returns:
        tuple: the average central, mean values of bins, standard deviation of bins.
    """
    bin_mean_rets = binned_statistic(pho.flatten(), data.flatten(), statistic='mean',
                                     bins=bins)
    bin_std_rets = binned_statistic(pho.flatten(), data.flatten(), statistic='std',
                                    bins=bins)
    radius_points = (bins[:-1] + bins[1:]) / 2
    return radius_points, bin_mean_rets[0], bin_std_rets[0]


def write_log(msg, file: TextIO):
    """
    Function help recording message.
    Parameters
    ----------
    msg : message that append to logfile
    file : file target

    Returns
    -------

    """
    print(msg)
    file.write(msg + '\n')
    return None


def assign_version(file_path, file_mode=True):
    """
    Add version tail in file path automatically.
    Parameters
    ----------
    file_path : str
        file path or dir path
    file_mode : bool
        if file mode is true, we only check the file version, i.e., not including directory version.

    Returns
    -------
    file_path : str
        new file path with version tail.

    """
    compiled_file_name = os.path.basename(file_path)
    compiled_dir_name = os.path.dirname(file_path)
    if compiled_dir_name == '':
        compiled_dir_name = os.getcwd()
    if file_mode:
        compiled_files = [file.name for file in os.scandir(compiled_dir_name) if file.is_file()]
    else:
        compiled_files = [file.name for file in os.scandir(compiled_dir_name)]

    comp_file_minor_v = 1
    temp_mame = f'{compiled_file_name}.{comp_file_minor_v}'
    while temp_mame in compiled_files:
        comp_file_minor_v += 1
        temp_mame = f'{compiled_file_name}.{comp_file_minor_v}'

    compiled_file_name = temp_mame

    file_path = os.path.join(compiled_dir_name, compiled_file_name)
    return file_path

if __name__ == '__main__':
# %% Discriminate cell state, using Gaussian-Mixture model
#%%
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

    # ==================== Parameters for Simulation
    compiled_path = r'/home/fulab/colony_agent_based_compiled/colony_RunBatch'
    input_path = r'/home/fulab/PycharmProjects/colony_agent_based_SSA/model_paras/ssa_in25'
    output_dir = r'/media/fulab/Data_Raid/sunhui_code_ret'
    source_dir = r'/home/fulab/PycharmProjects/colony_agent_based_SSA'
    field_file_flag = False
    core_number = 60
    green_ratio_threshold = 0.9
    batchSize = 50
    # ==================== Parameters for Simulation

# compile
    compiled_path = assign_version(compiled_path)
    input_filename = os.path.basename(input_path)
    compiled_file_name = os.path.basename(compiled_path)
    task_name = f'{input_filename}_{compiled_file_name}'  # task name
    output_dir_no_version = os.path.join(output_dir, task_name)
    output_dir = assign_version(output_dir_no_version)

    logfile = open(os.path.join(source_dir, 'logs', f'CellsMD3D_{task_name}.log'), 'a')
    write_log(f'[CellsMD3D {task_name}] -> Compiling files', logfile)
    compile_command = f'''g++ {os.path.join(source_dir, '*.cpp')} -fopenmp -O3 -o {compiled_path}'''
    write_log(compile_command, logfile)
    command = Popen(args=compile_command,
                    stdout=PIPE, universal_newlines=True, shell=True)
    write_log(f'[CellsMD3D {task_name}] -> Compiling, PID: {command.pid}', logfile)

    while True:
        output = command.stdout.readline()
        write_log(output, logfile)
        ret_code = command.poll()
        if ret_code is not None:
            for output in command.stdout.readlines():
                write_log(output, logfile)
            if ret_code == 0:
                write_log(f'[CellsMD3D {task_name}] -> Compiled finished.', logfile)
            else:
                write_log(f'[CellsMD3D {task_name}] -> Compiled failed. Exit code {ret_code}', logfile)
            break
        time.sleep(1)
    logfile.close()

    # start
    for batch_i in range(batchSize):
        output_dir = assign_version(output_dir_no_version, file_mode=False)

        if os.path.isdir(output_dir) is False:
            os.mkdir(output_dir)

        logfile = open(os.path.join(output_dir, f'CellsMD3D_{task_name}.log'), 'a')
        command_copy = f'''cp -f {input_path} {os.path.join(output_dir, task_name + '.txt')}'''
        write_log(f'[CellsMD3D {task_name}] -> Copy parameter file.', logfile)
        command = Popen(args=command_copy,
                        stdout=PIPE, universal_newlines=True, shell=True)
        cell_files_dir = os.path.join(output_dir, 'Cells')

        # Simulate the Colony model
        write_log(f'[CellsMD3D {task_name}] -> Start simulation.', logfile)
        if field_file_flag is False:
            field_file = '0'
        else:
            field_file = ''
        commands3 = f'''{compiled_path} {input_path} {core_number} {output_dir} {field_file}'''
        write_log(commands3, logfile)
        command = Popen(args=commands3,
                        stdout=PIPE, universal_newlines=True, shell=True)
        # morning cell states
        cells_file_num = 0
        ratio_list = []

        # ======== show the predict quality =============== #
        fig3, ax3 = plt.subplots(1, 1, figsize=(15, 15))
        ax3.scatter(test_ret[-2, 2], test_ret[-2, 1], c=test_cls, cmap='coolwarm', alpha=.1)
        ax3.set_xlim(1, 100)
        ax3.set_ylim(1, 200)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Red')
        ax3.set_ylabel('Green')
        splt.aspect_ratio(1)
        fig3.savefig(os.path.join(output_dir, 'Predict_standard.svg'))
        plt.close(fig3)
        while True:
            output = command.stdout.readline()
            write_log(output, logfile)
            wait_cells_data = True

            ret_code = command.poll()
            if ret_code is not None:
                if ret_code == 0:
                    write_log(f'[CellsMD3D {task_name}] -> Simulation finished.', logfile)
                else:
                    write_log(f'[CellsMD3D {task_name}] -> Simulation failed. Exit code {ret_code}', logfile)
                break  # Break this simulation loop and start the next loop.

            while wait_cells_data:
                try:
                    cell_files = os.listdir(cell_files_dir)
                    wait_cells_data = False
                    cell_files.sort(key=lambda name: int(name.split('.')[0]))
                except FileNotFoundError:
                    write_log(f'[CellsMD3D {task_name}] -> Waiting cells data.', logfile)
                    cell_files = []
                    time.sleep(5)

            if cells_file_num < len(cell_files):
                cells_file_num = len(cell_files)
                print(f'Loading file: {cell_files[-1]}')
                cells = read_cells_rets(os.path.join(cell_files_dir, cell_files[-1]))
                cells_all_location = np.array([cell.center for cell in cells])
                cells_all_R = np.array([cell.R for cell in cells])
                cells_all_G = np.array([cell.G for cell in cells])
                cells_all_lambda = np.array([cell.GrowthRate for cell in cells])
                cells_all_states = gmm.predict(np.log((cells_all_G + 1.) / (cells_all_R + 1.)).reshape(-1, 1))
                greenRatio = np.sum(cells_all_states == green_label) / len(cells)
                write_log(f'Green Ratio: {greenRatio}', logfile)
                ratio_list.append(greenRatio)
                fig4, ax4 = plt.subplots(1, 1, figsize=(15, 15))
                ax4.scatter(cells_all_R, cells_all_G, c=cells_all_states, cmap='coolwarm', alpha=.1)
                ax4.set_xlim(1, 100)
                ax4.set_ylim(1, 200)
                ax4.set_xscale('log')
                ax4.set_yscale('log')
                ax4.set_xlabel('Red')
                ax4.set_ylabel('Green')
                splt.aspect_ratio(1)
                fig4.savefig(os.path.join(output_dir, f'{cell_files[-1]}_population_predict.svg'))
                plt.close(fig4)
                z_top = 4
                z_bottom = -4
                location_mask = np.logical_and(cells_all_location[..., -1] > z_bottom,
                                               cells_all_location[..., -1] < z_top)

                cells_location = cells_all_location[location_mask, ...]
                cells_R = cells_all_R[location_mask]
                cells_G = cells_all_G[location_mask]
                cells_states = cells_all_states[location_mask]
                fig1_colony_bottom, ax1 = plt.subplots(1, 1, figsize=(15, 15))
                red_cells_loc = cells_location[cells_states != green_label]
                green_cells_loc = cells_location[cells_states == green_label]

                ax1.scatter(green_cells_loc[:, 0], green_cells_loc[:, 1], color=tuple(GreenColor), s=65, alpha=.6)
                ax1.scatter(red_cells_loc[:, 0], red_cells_loc[:, 1], color=tuple(RedColor), s=65, alpha=.6)
                ax1.set_xlim(-150, 150)
                ax1.set_ylim(-150, 150)

                fig1_colony_bottom.savefig(os.path.join(output_dir, f'{cell_files[-1]}_Colony_bottom.svg'))
                plt.close(fig1_colony_bottom)



                plt.close(fig4)
                if greenRatio > green_ratio_threshold:
                    write_log(f'[CellsMD3D {task_name}] -> Stop Simulation (Green ratio extend the green ratio).', logfile)
                    command.kill()
                    # logfile.close()
                    break
            time.sleep(5)

        write_log(f'[CellsMD3D {task_name}] -> Simulation Finish.', logfile)
        logfile.close()
