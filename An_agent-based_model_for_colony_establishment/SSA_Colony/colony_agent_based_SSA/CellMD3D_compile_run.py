import sys
import os
import getopt
import subprocess
import time
from typing import TextIO
import threading

def assign_version(file_path):
    compiled_file_name = os.path.basename(file_path)
    compiled_dir_name = os.path.dirname(file_path)
    if compiled_dir_name == '':
        compiled_dir_name = os.getcwd()

    compiled_files = [file.name for file in os.scandir(compiled_dir_name) if file.is_file()]
    comp_file_minor_v = 1
    temp_mame = f'{compiled_file_name}.{comp_file_minor_v}'
    while temp_mame in compiled_files:
        comp_file_minor_v += 1
        temp_mame = f'{compiled_file_name}.{comp_file_minor_v}'

    compiled_file_name = temp_mame

    file_path = os.path.join(compiled_dir_name, compiled_file_name)
    return file_path

def write_log(msg, file: TextIO):
    print(msg)
    file.write(msg + '\n')

def run_one_pip(compiled_path, input_path, output_path, core_number=32):

    compiled_path = assign_version(compiled_path)
    input_filename = os.path.basename(input_path)
    compiled_file_name = os.path.basename(compiled_path)
    task_name = f'{input_filename}_{compiled_file_name}'  # task name
    output_path = os.path.join(output_path, task_name)

    logfile = open(f'./logs/CellsMD3D_{task_name}.log', 'a')
    write_log(f'[CellsMD3D {task_name}] -> Compiling files', logfile)
    compile_command = f'''g++ {os.path.join(source_dir, '*.cpp')} -fopenmp -O3 -o {compiled_path}'''
    write_log(compile_command, logfile)
    command = subprocess.Popen(args=compile_command,
                               stdout=subprocess.PIPE, universal_newlines=True, shell=True)
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
                return 1
            break
        time.sleep(5)
    # Crate the directory for saving data
    commands2 = f'''mkdir -p {output_path}'''
    write_log(commands2, logfile)
    command2 = subprocess.Popen(args=commands2,
                                stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    # Copy the paramter file to save directory
    commd_copy = f'''cp -f {input_path} {os.path.join(output_path, task_name+'.txt')}'''
    write_log(f'[CellsMD3D {task_name}] -> Copy paramters.', logfile)
    command = subprocess.Popen(args=commd_copy,
                               stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    # Simulate the Colony model
    write_log(f'[CellsMD3D {task_name}] -> Start simulation.', logfile)
    if field_file_flag is False:
        field_file = '0'
    else:
        field_file = ''
    commands3 = f'''{compiled_path} {input_path} {core_number} {output_path} {field_file}'''
    write_log(commands3, logfile)
    command = subprocess.Popen(args=commands3,
                               stdout=subprocess.PIPE, universal_newlines=True, shell=True)

    while True:
        output = command.stdout.readline()
        write_log(output, logfile)
        ret_code = command.poll()
        if ret_code is not None:
            for output in command.stdout.readlines():
                write_log(output, logfile)
            if ret_code == 0:
                write_log(f'[CellsMD3D {task_name}] -> Finish Simulation.', logfile)
                logfile.close()
                return 0
            else:
                write_log(f'[CellsMD3D {task_name}] -> Simulation failed', logfile)
                logfile.close()
                return 1
        time.sleep(10)

# -n [compiled name] -

MultiP_flag = False
compiled_path = None  # file
input_path = None  # file
output_path = None  # dir
source_dir = os.path.dirname(sys.argv[0])  # dir
if source_dir == '':
    source_dir = os.getcwd()

core_number = 1  # defaults to using 1 cpu core.
# n: compile path; i: input file, parameter; o: output path; 
opts, arg = getopt.getopt(sys.argv[1:], 'n:i:o:h:f:', ['help', 'core_num='])





for opt, par in opts:
    if opt == '-n':
        compiled_path = par
    elif opt == '-i':
        input_path = par
    elif opt == '-o':
        output_path = par
    elif opt == '-f':
        lc_par = par.lower()
        if lc_par == 'true':
            field_file_flag = True
        else:
            field_file_flag = False

    elif opt in ['-h', '--help']:
        print('Help ! \n'
              '$> python CellMD3D_compile_run.py '
              '-n [path for compiled file] -i [path for parameter file] -o [path for out files]  -f [bool, field file]'
              '--core_num [core number] \n'
              'Example: \n'
              '$ python ./CellMD3D_compile_run.py -n ~/colony_agent_based_compiled/Colony_V0.3.7 -i ./model_paras/ssa_in25 -o /media/fulab/fulab_zc_1/sunhui_code_ret -f false --core_num 32')
        sys.exit(0)
    elif opt == '--core_num':
        core_number = par


if arg:
    MultiP_flag = True
    pars_list = arg
    threads_ist = []
    for pra_path in pars_list:
        threads_ist.append(threading.Thread(target=run_one_pip,
                                            args=(compiled_path, pra_path, output_path, core_number)))
    for pip_thread in threads_ist:
        pip_thread.start()
        time.sleep(5)
    print('[CellsMD3D] -> All tasks running !')
    for pip_thread in threads_ist:
        pip_thread.join()
    print('[CellsMD3D] -> All tasks finish !')
else:
    run_one_pip(compiled_path, input_path, output_path, core_number)



