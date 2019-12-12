#PBS -N cnn_mnist_gpu
#PBS -l walltime=0:30:00
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2
module load cuda/10.0.130
python -u get_training_data.py >& get_training_data_gpu.lg
