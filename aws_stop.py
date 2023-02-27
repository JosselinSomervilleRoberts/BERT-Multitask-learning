# Using the infos in config.ini, stops the EC2 instance

import configparser
from aws_utils import stop_instance

# config
config = configparser.ConfigParser()
config.read('config.ini')

# Start the EC2 instance
instance_id = config['AWS']['INSTANCE_ID']
region_name = config['AWS']['REGION']
stop_instance(instance_id, region_name)