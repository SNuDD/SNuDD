#!/bin/bash
#----------------------------------------------------
# SLURM job script with SBATCH
#----------------------------------------------------

#SBATCH -J Scan_Cluster                                            # Cluster job name
#SBATCH --get-user-env
#SBATCH -o /hydrarepo/vcosta/LOG/%x_%j.out                         # %x adds job name, %j is JobID
#SBATCH -e /hydrarepo/vcosta/LOG/%x_%j.err                         
#SBATCH --mail-type=FAIL                                           
#SBATCH --mail-user=valeria.costa@ift.csic.es
#SBATCH -N 1                                                       # Changed -n to -N (1 Node)
#SBATCH -c 32                                                      # Cores per task requested
#SBATCH --mem-per-cpu=1GB                                          # Memory per core
#SBATCH -t 2-00:00:00                                              # Run time (dd-hh:mm:ss)
#SBATCH --partition=batch4                                         # Queues on hydra: batch4, long

NCORES=$1
NSI_ELEM=$2

eval "$(conda shell.bash hook)"                                    
conda activate myfolder                                             

cd /hydrarepo/vcosta/SNuDD/  

# Run the python script with $NCORES threads and $NSI_ELEM
python Scan_Cluster.py $NCORES $NSI_ELEM                                       

echo "SBATCH: done"                                              
exit 0;