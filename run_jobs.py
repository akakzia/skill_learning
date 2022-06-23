import os
from time import sleep

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = os.environ['SCRATCH']

# Make top level directories
mkdir_p(job_directory)

envs = ['HandReach-v0', 'HandManipulateBlockRotateXYZ-v0']
nb_seeds = 1

for i in range(nb_seeds):
    for env in envs:
        job_file = os.path.join(job_directory, f"test_{env}%.slurm")

        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --account=kcr@v100\n")
            fh.writelines(f"#SBATCH --job-name=test_{env}\n")
            fh.writelines("#SBATCH --qos=qos_gpu-dev\n")
            fh.writelines(f"#SBATCH --output=test_{env}%_%j.out\n")
            fh.writelines(f"#SBATCH --error=test_{env}%_%j.out\n")
            fh.writelines("#SBATCH --time=19:59:59\n")
            fh.writelines("#SBATCH --ntasks=24\n")
            fh.writelines("#SBATCH --ntasks-per-node=1\n")
            fh.writelines("#SBATCH --gres=gpu:1\n")
            fh.writelines("#SBATCH --hint=nomultithread\n")
            fh.writelines("#SBATCH --array=0-0\n")

            fh.writelines("module load pytorch-gpu/py3/1.4.0\n")

            fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
            fh.writelines("export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
            fh.writelines("export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/include\n")
            fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uqy56ga/.mujoco/mujoco200/bin\n")
            fh.writelines("export OMPI_MCA_opal_warn_on_missing_libcuda=0\n")
            fh.writelines("export OMPI_MCA_btl_openib_allow_ib=1\n")
            fh.writelines("export OMPI_MCA_btl_openib_warn_default_gid_prefix=0\n")
            fh.writelines("export OMPI_MCA_mpi_warn_on_fork=0\n")

            fh.writelines(f"srun python -u -B train.py --env-name {env} --save-dir 'test_{env}/' 2>&1 ")

        os.system("sbatch %s" % job_file)
        sleep(1)
        