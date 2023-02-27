# Using the infos in config.ini, starts the EC2 instance

import configparser
from aws_utils import start_instance
import subprocess
import time

# config
config = configparser.ConfigParser()
config.read('config.ini')

# Start the EC2 instance
instance_id = config['AWS']['INSTANCE_ID']
region_name = config['AWS']['REGION']
start_instance(instance_id, region_name)

# Wait for the instance to start
print("Waiting 10 seconds for the instance to start...")
time.sleep(10)

# Connect to the EC2 instance and pull the latest code from GitHub
cmd_aws = "cd CS224N-Project-BERT-MultiTask/; git fetch; git checkout main; git pull;"
cmd_str = "ssh -i " + config['AWS']['SSH_KEY_PATH'] + " ec2-user@" + config['AWS']['PUBLIC_IP'] + " '" + cmd_aws + "'"
print("Running command: " + cmd_str)
subprocess.run(cmd_str, shell=True)

print("\nTo login, run the following command:")
print("ssh -t -i " + config['AWS']['SSH_KEY_PATH'] + " ec2-user@" + config['AWS']['PUBLIC_IP']
    + " 'cd CS224N-Project-BERT-MultiTask/; bash'")