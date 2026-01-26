import numpy as np
from f90nml import read

from spectre import SPECTREout, spec_fft_fast

def find_axis_from_pncare(fname, fname_write, plot=False):

    o = SPECTREout(fname+'.h5')

    nfp = o.input.physics.Nfp

    if plot:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 4, figsize=(13,5), sharey=True)

        for i in range(len(ax)):
            zeta = i/len(ax)

            o.plot_infaces(zetan=zeta, ax=ax[i], plot_axis=True)
            o.plot_poincare(zetan=zeta, ax=ax[i])
            if i>0:
                ax[i].set_ylabel('')
                ax[i].set_yticks([])

        plt.tight_layout()

    rarr = o.poincare.R[0]
    zarr = o.poincare.Z[0]

    rarr_avg = np.mean(rarr, axis=0)
    zarr_avg = np.mean(zarr, axis=0)

    zetas = np.linspace(0, 2 * np.pi/nfp, rarr_avg.shape[0], endpoint=False)

    reven, rodd, ims, ins = spec_fft_fast(np.array([0.]), zetas, np.array([rarr_avg]).T, 1, 3)
    zeven, zodd, ims, ins = spec_fft_fast(np.array([0.]), zetas, np.array([zarr_avg]).T, 1, 3)

    if plot:
        for i in range(len(ax)):
            zeta = i*(2*np.pi/nfp)/len(ax)

            ind = np.argmin((zetas-zeta)**2)
            ax[i].plot(rarr_avg[ind], zarr_avg[ind], 'rx', ms=4)

    nml = read(fname)
    for name, val in [('Rac',reven), ('Ras',rodd), ('Zac',zeven), ('Zas',zodd)]:
        print(name, val)
        temp = np.array(nml['physicslist'][name])
        temp[:4] = val
        nml['physicslist'][name] = temp
    nml.write(fname_write)