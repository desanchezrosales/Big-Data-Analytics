#PBS -N kfold_100k
#PBS -l walltime=25:30:00
#PBS -l nodes=1:ppn=40
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2
python -u kfold_100k.py >& kfold_100k.lg

