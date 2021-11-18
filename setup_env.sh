source ~/miniconda3/etc/profile.d/conda.sh

conda create -n env_gdl python=3.6 numpy pandas networkx matplotlib seaborn jupyterlab --yes
conda activate env_gdl

conda install pytorch=1.9 torchvision torchaudio cudatoolkit=10.2 -c pytorch -c nvidia --yes
conda install pyg -c pyg -c conda-forge --yes

pip install -U ray[tune]
pip install wandb black numba

cd gdl
python setup.py develop
cd ..
