import mlogger

def _plot_visdom(exp_name):
    new_plotter = mlogger.VisdomPlotter({'server': 'http://helios.robots.ox.ac.uk',
                                         'port': 9031,
                                         'env': '{}'.format(exp_name)})
    save_path = '/home/alasdair/stuff_to_plot'
    new_xp = mlogger.load_container(f'{save_path}/{exp_name}/results.json')
    new_xp.plot_on(new_plotter)
    new_plotter.update_plots()


def main():
    exp_names = [
        'wrn10-alig--k-2--eta-0.03--l2-None--b-120-25_epoch_pretrain--momentum-0.9',
        'wrn10-sbd-sb--k-3--eta-0.03--l2-None--b-120-25_epoch_pretrain--momentum-0.9',
        'wrn10-sbd-sb--k-5--eta-0.03--l2-None--b-30-25_epoch_pretrain--momentum-0.9',
    ]
    for exp_name in exp_names:
        _plot_visdom(exp_name)

if __name__ == '__main__':
    main()
