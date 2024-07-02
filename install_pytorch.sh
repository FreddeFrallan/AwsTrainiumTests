# Install Python venv 

sudo apt-get install -y python3.10-venv g++ 



# Create Python venv

python3.10 -m venv aws_neuron_venv_pytorch 



# Activate Python venv 
source aws_neuron_venv_pytorch/bin/activate 
python -m pip install -U pip 

# Install Jupyter notebook kernel
pip install ipykernel 
python3.10 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository 
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli 
python -m pip install wget 
python -m pip install awscli 

# Install Neuron Compiler and Framework
python -m pip install neuronx-cc==2.* torch-neuronx torchvision

# current working version requires transformers in a specific version
python -m pip install transformers==4.33.1

# install optimum package
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
python -m pip install optimum[neuronx]

echo "DONE!"