#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -J Colony_V0.10_in14
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=35

#python3 ~/bin/colony_agent_based/CellMD3D_compile_run.py -n ~/bin/colony_agent_based_compiled/Colony_V0.10 -i ~/bin/colony_agent_based/model_paras/in14 -o ~/bin/colony_agent_based_rets/ --core_num 36


 python3 ~/bin/colony_agent_based/CellMD3D_compile_run.py -n ~/bin/colony_agent_based_compiled/Colony_V0.12_in16_17 -o ~/bin/colony_agent_based_rets/ --core_num 36 ~/bin/colony_agent_based/model_paras/in16 ~/bin/colony_agent_based/model_paras/in17