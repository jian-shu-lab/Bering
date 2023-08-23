#BSUB -M 64000
#BSUB -W 2:00

module load anaconda3 
module load cuda/11.6 
source activate /data/aronow/Kang/py_envir/aipy2

make html