#PBS -N full_data_trial_32_layers
#PBS -l walltime=12:30:00
#PBS -l nodes=1:ppn=40
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2
python -u full_data_trial_32_layers.py >& full_data_trial_32_layers.lg
