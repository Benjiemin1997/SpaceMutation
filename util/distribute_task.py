import os
import math
import shutil
import torch
from concurrent.futures import ThreadPoolExecutor
import paramiko
from scp import SCPClient
from torchvision import datasets


servers = [
    {'host': 'xxx', 'username': 'xxx', 'password': 'xxx', 'path': 'xxx'},
    {'host': 'xxx', 'username': 'xxx', 'password': 'xxx', 'path': 'xxx'},
    # Add more server information
]

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)

# Calculate the amount of data allocated to each server
total_images = len(test_dataset)
num_servers = len(servers)
images_per_server = math.ceil(total_images / num_servers)

# Create a function to transfer the dataset to the server
def create_scp_client(server):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server['host'], username=server['username'], password=server['password'])
    return SCPClient(ssh.get_transport()), ssh


def transfer_to_server(server, dataset, indices):
    local_path = f'./temp/{server["host"]}'
    os.makedirs(local_path, exist_ok=True)
    print(f"Transferring {len(indices)} images to {server['host']}")
    for idx in indices:
        image, label = dataset[idx]
        target_path = os.path.join(local_path, f'{idx}.pt')
        torch.save((image, label), target_path)

    scp, ssh = create_scp_client(server)
    remote_path = server['path']
    ssh.exec_command(f'mkdir -p {remote_path}')

    scp.put(local_path, recursive=True, remote_path=remote_path)

    # Clean up local temporary files
    shutil.rmtree(local_path)
    scp.close()
    ssh.close()


def parallel_transfer(servers, dataset, images_per_server):
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        futures = []
        for i, server in enumerate(servers):
            start_idx = i * images_per_server
            end_idx = min((i + 1) * images_per_server, len(dataset))
            indices = list(range(start_idx, end_idx))
            futures.append(executor.submit(transfer_to_server, server, dataset, indices))

        for future in futures:
            future.result()



parallel_transfer(servers, test_dataset, images_per_server)