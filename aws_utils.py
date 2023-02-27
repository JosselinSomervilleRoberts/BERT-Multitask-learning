#!/usr/bin/env python
# coding: utf-8
import boto3
import os
import configparser
import json

config = configparser.ConfigParser()
config.read('config.ini')

def get_ec2_resource(region_name="us-east-1"):
    ec2_resource = boto3.resource("ec2",
                                    region_name=region_name,
                                    aws_access_key_id=config['AWS']['ACCESS_KEY'],
                                    aws_secret_access_key=config['AWS']['SECRET_KEY'])
    return ec2_resource

def get_ec2_client(region_name="us-east-1"):
    ec2_client = boto3.client("ec2",
                                region_name=region_name,
                                aws_access_key_id=config['AWS']['ACCESS_KEY'],
                                aws_secret_access_key=config['AWS']['SECRET_KEY'])
    return ec2_client

def create_key_pair_if_not_exists(key_name="ec2-key-pair", region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    key_pairs = ec2_client.describe_key_pairs()["KeyPairs"]
    key_pair_names = [key_pair["KeyName"] for key_pair in key_pairs]
    if key_name not in key_pair_names:
        path = create_key_pair(key_name, region_name)
        return path
    else:
        print("Key pair already exists. Trying to load it from file.")
        path = "aws_" + key_name + ".pem"
        if os.path.exists(path):
            print("Key pair file exists. Loading it.")
            return path
        else:
            raise Exception("Key pair file does not exist. Please create it manually.")

def create_key_pair(key_name="ec2-key-pair", region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    key_pair = ec2_client.create_key_pair(KeyName=key_name)

    private_key = key_pair["KeyMaterial"]

    # write private key to file with 700 permissions
    path = "aws_" + key_name + ".pem"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT, 0o700), "w+") as handle:
        handle.write(private_key)
    return path

def get_public_ip(instance_id, region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    reservations = ec2_client.describe_instances(InstanceIds=[instance_id]).get("Reservations")

    ip_addresses = []
    for reservation in reservations:
        for instance in reservation['Instances']:
            ip_addresses.append(instance.get("PublicIpAddress"))
    return ip_addresses

def start_instance(instance_id, region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    response = ec2_client.start_instances(InstanceIds=[instance_id])
    print(response)

def stop_instance(instance_id, region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    response = ec2_client.stop_instances(InstanceIds=[instance_id])
    print(response)
          
def terminate_instance(instance_id, region_name="us-east-1"):
    ec2_client = get_ec2_client(region_name)
    response = ec2_client.terminate_instances(InstanceIds=[instance_id])
    print(response)

def save_instance_ips(instance_ips):
    with open("ips.json", 'w') as f:
        f.write(json.dumps(instance_ips))

def load_instance_ips():
    with open("ips.json", 'r') as f:
        instance_ips = json.loads(f.read())
    return instance_ips