import argparse
import os
from PIL import Image
import numpy as np

import data
import model
import utils

def main():
    parser = argparse.ArgumentParser(description='Set config and checkpoint.')
    parser.add_argument('--config', metavar='str', required=True, type=str)
    parser.add_argument('--checkpoint', metavar='file', type=str, default='')
    args = parser.parse_args()
    config = utils.load_config(args.config)

    save_dir = os.path.join('checkpoints', config.exp_name, 'nll')
    os.makedirs(save_dir, exist_ok=True)

    test_loader = data.build_dataloader(config, phase='train')

    trainer = model.get_model(config)
    trainer.load_checkpoint(args.checkpoint)

    counts = {k: 0 for k in test_loader.dataset.classes}
    for feeddict in test_loader:
        if all([counts[k] > 50 for k in counts]):
            break
        feeddict['img'] = feeddict['img'].cuda()
        nll = trainer.compute_nll(feeddict)['total_bpd']
        nll = nll.cpu().numpy()
        nll[nll > 100] = 100
        nll = nll / 100 * 255
        nll = nll.astype(np.uint8)

        img = feeddict['img'].cpu().numpy()
        img = (img * 255).astype(np.uint8)

        for i in range(img.shape[0]):
            label = int(feeddict['label'][i, 0].item())
            label_txt = test_loader.dataset.classes[label]
            if counts[label_txt] > 50:
                continue
            # temp = Image.fromarray(img[i, 0])
            # temp.save(os.path.join('checkpoints', config.exp_name, f'train_sample_{i}.jpg'))
            temp = Image.fromarray(np.concatenate((img[i, 0], nll[i, 0]), axis=1))
            temp.save(os.path.join(save_dir, f'{label_txt}_sample_{counts[label_txt]}.jpg'), )
            counts[label_txt] += 1
        

if __name__ == '__main__':
    main()