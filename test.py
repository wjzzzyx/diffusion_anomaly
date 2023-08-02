import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from typing import List, Dict

import diffusion_anomaly
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', metavar='dir', type=str)
    parser.add_argument('--nll_timesteps', type=list, default=[600, 700, 800, 900, 950, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999])
    args = parser.parse_args()
    config = utils.load_config(os.path.join(args.checkpoint, 'config.yaml'))

    test_dataset = diffusion_anomaly.data.build_dataset(config, phase='test')
    test_loader = diffusion_anomaly.data.build_dataloader(config, test_dataset, phase='test')

    trainer = diffusion_anomaly.model.get_model(config)
    trainer.load_checkpoint(os.path.join(args.checkpoint, 'step100000.pth'))
    model = trainer.ema.ema_model

    records = list()
    for i, feeddict in enumerate(test_loader):
        label = feeddict['label']

        result = model.calc_bpd_loop(feeddict['img'].cuda(), timesteps=args.nll_timesteps)
        total_bpd = result['total_bpd'][0]
        prior_bpd = result['prior_bpd'][0]
        vb = result['vb'][0]
        pred_xstart = result['pred_xstart'][0]
        xstart_mse = result['xstart_mse'][0]

        # what metrics can we use to estimate anomaly?
        # mean, max, median, std, 95%, 90%
        vb_mean = torch.mean(vb, dim=(1, 2, 3))
        vb_max = torch.amax(vb, dim=(1, 2, 3))
        vb_std = torch.std(vb, dim=(1, 2, 3))
        vb_percentiles = torch.quantile(vb.flatten(1), torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 0.95], device=vb.device), dim=1)

        record = {
            'label': label[0].cpu().numpy(),
            'vb_mean': vb_mean.cpu().numpy(),
            'vb_max': vb_max.cpu().numpy(),
            'vb_std': vb_std.cpu().numpy(),
            'vb_p50': vb_percentiles[0].cpu().numpy(),
            'vb_p60': vb_percentiles[1].cpu().numpy(),
            'vb_p70': vb_percentiles[2].cpu().numpy(),
            'vb_p80': vb_percentiles[3].cpu().numpy(),
            'vb_p90': vb_percentiles[4].cpu().numpy(),
            'vb_p95': vb_percentiles[5].cpu().numpy(),
        }
        records.append(record)

    analyze(records)


def analyze(records: List[Dict]):
    num_classes = len(records[0]['label'])
    perclass_stats = {k: dict() for k in range(num_classes)}
    for record in records:
        classes = np.nonzero(record['label'])[0].tolist()
        for icls in classes:
            perclass_stats[icls].setdefault('vb_mean', []).append(record['vb_mean'])
            perclass_stats[icls].setdefault('vb_max', []).append(record['vb_max'])
            perclass_stats[icls].setdefault('vb_std', []).append(record['vb_std'])
            perclass_stats[icls].setdefault('vb_p50', []).append(record['vb_p50'])
            perclass_stats[icls].setdefault('vb_p60', []).append(record['vb_p60'])
            perclass_stats[icls].setdefault('vb_p70', []).append(record['vb_p70'])
            perclass_stats[icls].setdefault('vb_p80', []).append(record['vb_p80'])
            perclass_stats[icls].setdefault('vb_p90', []).append(record['vb_p90'])
            perclass_stats[icls].setdefault('vb_p95', []).append(record['vb_p95'])
    
    for icls in range(num_classes):
        for stat in perclass_stats[icls].keys():
            perclass_stats[icls][stat] = np.array(perclass_stats[icls][stat])

    for icls in range(1, num_classes):
        for stat in perclass_stats[0].keys():
            for step in range(perclass_stats[0][stat].shape[1]):
                dist1 = perclass_stats[0][stat][:, step]
                dist2 = perclass_stats[icls][stat][:, step]
                bins = np.linspace(0, max(np.max(dist1), np.max(dist2)), 100)
                plt.hist(dist1, bins, alpha=0.5, label='normal')
                plt.hist(dist2, bins, alpha=0.5, label=f'class{icls}')
                plt.legend(loc='upper right')
                plt.savefig(f'class{icls}_{stat}_step{step}.png')
                plt.clf()


if __name__ == '__main__':
    main()