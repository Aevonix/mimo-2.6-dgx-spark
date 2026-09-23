#!/usr/bin/env python3
"""Print or start one rank. Does not SSH, stop services, or change host settings."""
import argparse
import ipaddress
import json
from pathlib import Path
import re
import shlex
import subprocess


def command(config, rank, native=False):
    hosts = config['hosts']
    if len(hosts) != 8 or len(set(hosts)) != 8 or rank not in range(8):
        raise ValueError('This recipe requires eight unique hosts and rank 0..7')
    for host in hosts:
        ipaddress.ip_address(host)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', config['container_prefix']):
        raise ValueError('Invalid container prefix')
    for key in ['model_path', 'cache_path']:
        if not Path(config[key]).is_absolute() or any(c in config[key] for c in ',\n\r\0'):
            raise ValueError(f'{key} must be an absolute local path without mount separators')
    env = dict(config['environment'], VLLM_HOST_IP=hosts[rank])
    env['NCCL_IB_HCA'] = '=' + ','.join(hca + ':1' for hca in config['rdma_devices'])
    env['NCCL_IB_GID_INDEX'] = str(config['gid_index'])
    for key in ['NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME', 'TP_SOCKET_IFNAME', 'MN_IF_NAME']:
        env[key] = config['socket_interface']
    name = f"{config['container_prefix']}-r{rank}"
    args = ['docker', 'run', '-d', '--name', name, '--restart', 'no', '--gpus', 'all',
            '--network', 'host', '--ipc', 'host', '--shm-size', '32g',
            '--ulimit', 'memlock=-1:-1', '--cap-add', 'IPC_LOCK',
            '--device', '/dev/infiniband:/dev/infiniband',
            '--log-driver', 'local', '--log-opt', 'max-size=20m', '--log-opt', 'max-file=3',
            '--entrypoint', 'vllm',
            '--mount', f"type=bind,src={config['model_path']},dst=/models/mimo,readonly",
            '--mount', f"type=bind,src={config['cache_path']},dst=/cache"]
    for key, value in sorted(env.items()):
        args.extend(['--env', f'{key}={value}'])
    args.extend([config['image'], 'serve', '/models/mimo', '--served-model-name', 'mimo-v2.6-pro-rl',
                 '--host', hosts[0], '--port', str(config['api_port']), '--tensor-parallel-size', '8',
                 '--pipeline-parallel-size', '1', '--decode-context-parallel-size', '1',
                 '--nnodes', '8', '--node-rank', str(rank), '--master-addr', hosts[0],
                 '--master-port', str(config['master_port'])])
    base = config['baseline']
    for flag, key in [('max-model-len', 'max_model_len'), ('max-num-seqs', 'max_num_seqs'),
                      ('max-num-batched-tokens', 'max_num_batched_tokens'),
                      ('gpu-memory-utilization', 'gpu_memory_utilization'), ('kv-cache-dtype', 'kv_cache_dtype')]:
        args.extend(['--' + flag, str(base[key])])
    if not native:
        args.extend(['--speculative-config', json.dumps(base['speculative_config'])])
    args.extend(config['serve_arguments'])
    if rank:
        args.append('--headless')
    return args


def preflight(config, rank):
    if 'REPLACE_' in json.dumps(config) or any(ipaddress.ip_address(h) in ipaddress.ip_network('192.0.2.0/24') for h in config['hosts']):
        raise ValueError('Edit the example network configuration before starting')
    model = Path(config['model_path'])
    for relative in ['config.json', 'tokenizer_config.json', 'model.safetensors.index.json', 'dflash/config.json']:
        if not (model / relative).is_file():
            raise ValueError(f'Missing model file: {relative}')
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    if not all((model / path).is_file() for path in set(index['weight_map'].values())):
        raise ValueError('Model snapshot is incomplete')
    addresses = json.loads(subprocess.check_output(['ip', '-j', 'address', 'show', 'dev', config['socket_interface']], text=True))
    if not any(a.get('local') == config['hosts'][rank] for row in addresses for a in row['addr_info']):
        raise ValueError('Configured rank address is not on this host/interface')
    for hca in config['rdma_devices']:
        port = Path('/sys/class/infiniband') / hca / 'ports/1'
        if 'ACTIVE' not in (port / 'state').read_text():
            raise ValueError(f'RDMA device is not active: {hca}')
        if 'RoCE v2' not in (port / 'gid_attrs/types' / str(config['gid_index'])).read_text():
            raise ValueError(f'GID is not RoCE v2 on {hca}')
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if busy:
        raise ValueError('GPU already has a compute process; inspect it before starting')
    Path(config['cache_path']).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('cluster.json'))
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--native', action='store_true', help='Disable DFlash, retaining identical numerical overlays')
    parser.add_argument('--execute', action='store_true', help='Run this rank after local preflight; default prints the command')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cmd = command(config, args.rank, args.native)
    if args.execute:
        preflight(config, args.rank)
        subprocess.run(cmd, check=True)
    else:
        print(shlex.join(cmd))
