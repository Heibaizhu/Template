import os
import glob
import argparse
import subprocess
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='+')
    parser.add_argument('--port', type=str, default='7001')
    parser.add_argument('--len', type=int, default=0)
    args = parser.parse_args()
    return args



class MulTensorboard:
    def __init__(self):
        pass

    def logdir_spec(self, logdirs, port='7000'):

        logs = ""

        for key, value in logdirs.items():
            logs = logs + '{}:"{}",'.format(key, value)

        logs = logs.strip(',')
        cmd = 'tensorboard --logdir_spec=' + logs + '  --port {}'.format(port)
        print(cmd)
        subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    opt = parse_args()
    log_parser = MulTensorboard()
    log = {}
    for name in opt.name:
        log_dirs = './experiments/{}/tensorboard'.format(name)
        logs = sorted(glob.glob(os.path.join(log_dirs, '*')), reverse=False)
        log_len = opt.len if opt.len != 0 else len(logs)
        logs = {os.path.basename(log): log for log in logs[-log_len:]}
        log.update(logs)
    log_parser.logdir_spec(log, port=opt.port)