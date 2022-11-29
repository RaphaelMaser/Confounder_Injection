#!/bin/bash -l
#SBATCH -J "Hyperparams full finetuning"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128                # Cores assigned to each tasks
#SBATCH --time=0-36:00:00
#SBATCH -p batch
#SBATCH --array=0-9
#SBATCH --qos normal

samples=128
pbt=1
epochs=10000
sleep 3s
test_samples=8

module load lang/Anaconda3/2020.11
source activate confounder_3.10
d="$(date)"


case ${SLURM_ARRAY_TASK_ID} in

0)
#python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 0 -test_samples=$test_samples -target_domain_samples 0 -target_domain_confounding 0 -de_correlate_confounder_target 0 -de_correlate_confounder_test 0 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt
;;

1)
python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 0 -test_samples=$test_samples -target_domain_samples 2 -target_domain_confounding 0 -de_correlate_confounder_target 0 -de_correlate_confounder_test 0 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt 
;;

2)
python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 0 -test_samples=$test_samples -target_domain_samples 4 -target_domain_confounding 0 -de_correlate_confounder_target 0 -de_correlate_confounder_test 0 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt
;;

3)
#python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 1 -test_samples=$test_samples -target_domain_samples 0 -target_domain_confounding 1 -de_correlate_confounder_target 1 -de_correlate_confounder_test 1 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt
;;

4)
python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 1 -test_samples=$test_samples -target_domain_samples 2 -target_domain_confounding 1 -de_correlate_confounder_target 1 -de_correlate_confounder_test 1 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt
;;

5)
python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 1 -test_samples=$test_samples -target_domain_samples 4 -target_domain_confounding 1 -de_correlate_confounder_target 1 -de_correlate_confounder_test 1 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt
;;

6)
python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 1 -test_samples=$test_samples -target_domain_samples 8 -target_domain_confounding 1 -de_correlate_confounder_target 1 -de_correlate_confounder_test 1 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt 
;;

7)
python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 1 -test_samples=$test_samples -target_domain_samples 16 -target_domain_confounding 1 -de_correlate_confounder_target 1 -de_correlate_confounder_test 1 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt 
;;

8)
python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 1 -test_samples=$test_samples -target_domain_samples 32 -target_domain_confounding 1 -de_correlate_confounder_target 1 -de_correlate_confounder_test 1 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt 
;;

9)
python "BrNet hyperparams sequentiell.py" -finetuning 1 -test_confounding 1 -test_samples=$test_samples -target_domain_samples 64 -target_domain_confounding 1 -de_correlate_confounder_target 1 -de_correlate_confounder_test 1 -c ${SLURM_CPUS_PER_TASK} -d "$current_date" -epochs $epochs -samples $samples -pbt $pbt 
;;

esac

