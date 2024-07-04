curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install


source ~/aws_neuron_venv_pytorch/bin/activate
pip install transformers==4.31.0 tensorboard datasets


# put proper config into ~/.aws/config

# aws sso login

aws s3 cp --recursive s3://ekrakma/pp8_tp8/ pp8_tp8    # dsk33b-base model sharded for pp8 tp8

# copy or generate the data!



