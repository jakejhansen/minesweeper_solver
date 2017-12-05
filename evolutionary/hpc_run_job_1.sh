#!/bin/bash
#BSUB -J deep-evo-1
#BSUB -q hpc
#BSUB -W 10:30
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o deep-evo-1_%J.log

##!/bin/sh
##PBS -N deep-evo
##PBS -q hpc
##PBS -l walltime=01:30:00
##PBS -l nodes=1:ppn=8
##PBS -l vmem=6gb
##PBS -j oe
##PBS -o deep-evo.log 
##PBS -q k40_interactive

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


# load modules - do this manually beforehand! 
# is automatically done in new shells after running hpc_config_shell.sh on your user account
# module load python3
# module load ffmpeg/3.4
#export PYTHONPATH=~/stdpy3
#chmod 777 $PYTHONPATH/bin/activate
source stdpy3/bin/activate


#
# Start time
#
start_time=`date +%s`

export KERAS_BACKEND=theano
python3 "deep/evo/CartPole-v1-(4)/es-multi-threaded.py" --nwrk 1 > "deep/evo/CartPole-v1-(4)/output_001.txt"


# DONE
end_time=`date +%s`
run_time=$((end_time-start_time))

printf '\nScript finished. Took: %dh:%dm:%ds\n' \
  $(($run_time/3600)) $(($run_time%3600/60)) $(($run_time%60))
