import argparse

from bmtk.simulator import dpointnet
import matplotlib.pyplot as plt

import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt


def run(config_path):
    config = dpointnet.Config.from_json(config_path)
    config.build_env()

    rnn_network = dpointnet.RNN.from_config(config)
    results = rnn_network.run()
    fig = results.spikes.raster(batch_nums=0, show=False)
    fig.suptitle(f'Untrained results, firing_rate = {results.spikes.mean_firing_rate()}')
    fig.tight_layout()
    plt.show()
    # plt.close('all')

    # dpointnet.reset()

    # config = dpointnet.Config.from_json('config.train.all.json')
    # config.build_env()

    # rnn_network = dpointnet.RNN.from_config(config)
    # results = rnn_network.run()
    # fig = results.spikes.raster(batch_nums=0, show=False)
    # # untrained_results.spikes.raster(batch_nums=[0, 10], show=False)
    # fig.suptitle(f'Untrained results, firing_rate = {results}')
    # fig.tight_layout()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        type=str, 
        nargs='?', 
        default='config.inference.json'
    )

    args, _ = parser.parse_known_args()
    run(args.config_path)
