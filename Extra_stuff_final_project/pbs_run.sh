#PBS -N cnn_mnist
#PBS -l walltime=6:30:00
#PBS -l nodes=1:ppn=8
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2
python -u get_training_data.py >& get_training_data.lg
