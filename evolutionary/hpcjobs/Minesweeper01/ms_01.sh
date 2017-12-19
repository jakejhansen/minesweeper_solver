#!/bin/bash
#BSUB -J deep-evo
#BSUB -q hpc
#BSUB -W 10:30
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o deep-evo_%J.log

##!/bin/sh
##PBS -N deep-evo
##PBS -q hpc
##PBS -l walltime=01:30:00
##PBS -l nodes=1:ppn=8
##PBS -l vmem=6gb
##PBS -j oe
##PBS -o deep-evo.log 

## source https://github.com/AndreasMadsen/my-setup/blob/master/dtu-hpc-python3/setup-python3.sh
## copy to hpc using
## rsync -t hpc_run_job.sh s123249@transfer.gbar.dtu.dk:deep/evo/hpc_run_job.sh
## Idea for MPI: mpirun -np 1 --bind-to none "deep/evo/code.py"


# Stop on error
set -e

# Set $HOME if running as a qsub script
if [ -z "$BSUB_O_WORKDIR" ]; then
    export HOME=$BSUB_O_WORKDIR
fi

cd $HOME

export KERAS_BACKEND=theano

# load modules - do this manually beforehand! 
# is automatically done in new shells after running hpc_config_shell.sh on your user account
module load python3/3.6.2
module load scipy/0.19.1-python-3.6.2
module load matplotlib/2.0.2-python-3.6.2
source stdpy362/bin/activate

#
# Start time
#
start_time=`date +%s`

python3 Documents/deep/evolutionary/hpcjobs/Minesweeper01/run-minesweeper.py --nwrk 8 --nags 60 --ngns 30000 --cint 25 --sigm 0.1 --lrte 0.1 --regu 0.003 --size 6 --mine 7 > Documents/deep/evolutionary/hpcjobs/Minesweeper01/out01.txt


# DONE
end_time=`date +%s`
run_time=$((end_time-start_time))

printf '\nScript finished. Took: %dh:%dm:%ds\n' \
  $(($run_time/3600)) $(($run_time%3600/60)) $(($run_time%60))
