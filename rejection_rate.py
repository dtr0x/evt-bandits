import numpy as np

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(threshold=10000)

    burr_cvars = np.load('data/burr_cvars.npy')
    frec_cvars = np.load('data/frec_cvars.npy')
    t_cvars = np.load('data/t_cvars.npy')
    lnorm_cvars = np.load('data/lnorm_cvars.npy')
    weib_cvars = np.load('data/weib_cvars.npy')

    cvars_all = [burr_cvars, frec_cvars, t_cvars, lnorm_cvars, weib_cvars]

    chosen_tp = np.asarray([c[2] for c in cvars_all])
    rejection_rate = np.asarray([c[3] for c in cvars_all])/20

    avg_tp = np.nanmean(chosen_tp, axis=2)
    avg_rr = np.nanmean(rejection_rate, axis=2)

    i = 0
    for dist_class in rejection_rate:
        for dist_data in dist_class:
            print(i, np.around(np.where(dist_data == 1)[0].size/dist_data.size * 100, 3))
            i += 1
