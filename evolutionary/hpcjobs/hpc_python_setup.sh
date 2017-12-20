#!/bin/sh
## source https://github.com/AndreasMadsen/my-setup/blob/master/dtu-hpc-python3/setup-python3.sh
## copy to hpc using
## rsync -t hpc_python_setup.sh s123249@transfer.gbar.dtu.dk:deep/evo/hpc_python_setup.sh
#BSUB -J setup
#BSUB -q hpc
#BSUB -W 01:30
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o setup_%J.log

# Stop on error
set -e

# Set $HOME if running as a qsub script
if [ -z "$PBS_O_WORKDIR" ]; then
    export HOME=$PBS_O_WORKDIR
fi

# load modules
module load python3/3.6.2
module load scipy/0.19.1-python-3.6.2
module load matplotlib/2.0.2-python-3.6.2

# Use HOME directory as base
cd $HOME

#
# Start time
#
start_time=`date +%s`

#
# Setup virtual env
#	
rm -rf ~/stdpy3
python3 -m venv ~/stdpy3 --copies
source ~/stdpy3/bin/activate

#
# Install python modules
#
pip3 install keras IPython joblib gym tensorflow h5py theano mss

# DONE
end_time=`date +%s`
run_time=$((end_time-start_time))

printf '\nInstall script finished. Took: %dh:%dm:%ds\n' \
  $(($run_time/3600)) $(($run_time%3600/60)) $(($run_time%60))
