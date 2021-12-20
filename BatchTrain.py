import os
import subprocess
import time


if __name__ == '__main__':
    ymls = [
            'toy1.yml',
            'toy2.yml',
            # 'toy3.yml',
            # 'toy4.yml'
             ]

    for yml in ymls:
        c_path = os.path.join('./options/train/', yml)
        assert os.path.exists(os.path.join('./options/train', yml)), c_path

    cmd = 'OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1,7 python -m torch.distributed.launch --nproc_per_node=2 train.py --opt '
    for yml in ymls:
        cur_cmd = cmd + os.path.join('./options/train', yml)
        print(cur_cmd)
        try:
            subprocess.run(cur_cmd, shell=True)
            time.sleep(60)
        except:
            # pass
            time.sleep(60)
        # time.sleep(180)