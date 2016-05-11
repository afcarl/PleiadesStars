import numpy as np
from george import GP, kernels
import emcee, corner
import pylab as pl
from argparse import ArgumentParser
import os.path
import matplotlib.gridspec as gridspec

def readLC(file):
    X = np.genfromtxt(file).T
    t, y, ye = X
    ym = y.mean()
    y = y / ym
    ye = ye / ym
    return t, y, ye

def lnlike_QP1(p, t, y):
    ln_a, ln_tau_p, period, ln_tau_e_rel, ln_sig = p
    a, tau_p, tau_e_rel, sig = \
      np.exp(ln_a), np.exp(ln_tau_p), np.exp(ln_tau_e_rel), np.exp(ln_sig)
    a2 = a**2
    gamma_p = 1. / 2. / tau_p**2
    tau_e = tau_e_rel * period
    metric_e = tau_e**2
    gp = GP(a * kernels.ExpSine2Kernel(gamma_p, period) * \
            kernels.ExpSquaredKernel(metric_e))
    a, tau = np.exp(p[:2])
    yerr = np.ones(len(y)) * sig
    gp.compute(t, yerr)
    return gp.lnlikelihood(y)
    
def lnprior_QP1(p, per_init):
    """Jeffreys priors all round except for period where it's uniform"""
    ln_a, ln_tau_p, period, ln_tau_e_rel, ln_sig = p
    if (ln_a < -7) or (ln_a > 1):
        return -np.inf
    if (ln_tau_p < -1.4) or (ln_tau_p > 2.3):
        return -np.inf
    if (period < (0.5 * per_init)) or (period > (2.0 * per_init)):
        return -np.inf
    if (ln_tau_e_rel < 0.0) or (ln_tau_e_rel > 4.0):
        return -np.inf    
    if (ln_sig < -9) or (ln_sig > -2.3):
        return -np.inf
    return 0.0

def lnprob_QP1(p, t, y, per_init):
    lp = lnprior_QP1(p, per_init)
    if np.isfinite(lp):
        ll = lnlike_QP1(p, t, y)
        return lp + ll
    else:
        return -np.inf

def plotchains(sampler, labels):
    samples = sampler.chain
    lnp = sampler.lnprobability
    nwalkers, nsteps, ndim = samples.shape
    fig = pl.figure(figsize=(8, ndim + 1))
    gs = gridspec.GridSpec(ndim+1, 1)
    gs.update(left=0.15, right=0.98, bottom = 0.07, top = 0.93, hspace=0.01)
    ax1 = pl.subplot(gs[0,0])    
    ax1.yaxis.set_major_locator(pl.MaxNLocator(5, prune = 'both'))
    pl.setp(ax1.get_xticklabels(), visible=False)
    pl.ylabel(r"$\log \, \mathrm{posterior}$")
    for j in range(nwalkers):
        pl.plot(lnp[j,:], color="#4682b4", alpha=0.3)
    for i in range(ndim):
        axc = pl.subplot(gs[i+1,0])    
        axc.yaxis.set_major_locator(pl.MaxNLocator(5, prune = 'both'))
        if i < (ndim-1):
            pl.setp(axc.get_xticklabels(), visible=False)
        pl.ylabel(labels[i])
        for j in range(nwalkers):
            pl.plot(samples[j,:,i], color="#4682b4", alpha=0.3)
        if i == (ndim-1):
            pl.xlabel('iteration')
    pl.show(block=False)
    ans = int(raw_input('Enter number of iterations to discard as burn-in: '))
    for i in range(ndim+1):
        axc = pl.subplot(gs[i,0])
        pl.axvline(ans, color="#4682b4")
    print 'Discarding first %d steps as burn-in' % ans
    return samples[:,ans:,:]
                     
def plotsample_QP1(p, t, y):
    ln_a, ln_tau_p, period, ln_tau_e_rel, ln_sig = p
    a, tau_p, tau_e_rel, sig = \
      np.exp(ln_a), np.exp(ln_tau_p), np.exp(ln_tau_e_rel), np.exp(ln_sig)
    a2 = a**2
    gamma_p = 1. / 2. / tau_p**2
    tau_e = tau_e_rel * period
    metric_e = tau_e**2
    gp = GP(a * kernels.ExpSine2Kernel(gamma_p, period) * \
            kernels.ExpSquaredKernel(metric_e))
    a, tau = np.exp(p[:2])
    yerr = np.ones(len(y)) * sig
    gp.compute(t, yerr)
    mu = gp.sample_conditional(y, t)
    pl.plot(t, mu, color="#4682b4", alpha=0.3)

def plotpred_QP1(p, t, y):
    ln_a, ln_tau_p, period, ln_tau_e_rel, ln_sig = p
    a, tau_p, tau_e_rel, sig = \
      np.exp(ln_a), np.exp(ln_tau_p), np.exp(ln_tau_e_rel), np.exp(ln_sig)
    a2 = a**2
    gamma_p = 1. / 2. / tau_p**2
    tau_e = tau_e_rel * period
    metric_e = tau_e**2
    gp = GP(a * kernels.ExpSine2Kernel(gamma_p, period) * \
            kernels.ExpSquaredKernel(metric_e))
    a, tau = np.exp(p[:2])
    yerr = np.ones(len(y)) * sig
    gp.compute(t, yerr)
    mu, cov = gp.predict(y, t)
    sig = np.diag(cov)
    pl.fill_between(t, mu + 2 * sig, mu - 2 * sig, color="#4682b4", alpha=0.3)
    pl.plot(t, mu,  color="#4682b4", lw = 2)
    
if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('infile', metavar='IF', type=str, help='input file')
    ap.add_argument('period', metavar='P', type=float, help='initial period estimate')
    ap.add_argument('--output_dir', metavar='OD', type=str, \
                    default='.', help='directory to store output in')
    ap.add_argument('--nsteps', metavar='NS', type=int, default=1000, \
                    help='number of MCMC steps')
    ap.add_argument('--nwalkers', metavar='NW', type=int, default=24, \
                    help='number of MCMC walkers')
    ap.add_argument('--force-mcmc', action='store_true', default=False, \
                    help='re-run MCMC even if results file exists')
    args = ap.parse_args()

    t, y, ye = readLC(args.infile)
    sig = (y[1:] - y[:-1]).std() / np.sqrt(2)
    h = y.var()
    ye[:] = sig
    
    pl.figure(1)
    pl.errorbar(t, y, yerr = ye, fmt = 'k.', capsize = 0)
    pl.xlabel('time (d)')
    pl.ylabel('norm. flux')
    fname = os.path.splitext(os.path.split(args.infile)[-1])[0]
    dfname = '%s/%s' % (args.output_dir, fname)
    pl.title(fname)

    # do MCMC
    mcmc_output_file = '%s_QPGP_chain.dat' % dfname
    if (os.path.exists(mcmc_output_file) == False) or (args.force_mcmc == True):
        print 'Doing MCMC'
        p_init = np.array([-3.0, 0.5, args.period, 1.15, -5.65])
        ndim = len(p_init)
        p0 = [np.array(p_init) + 1e-8 * np.random.randn(ndim) \
            for i in xrange(args.nwalkers)]
        ss = 1
        dt = np.median(t[1:] - t[:-1])
        while ((20 * dt * ss) < args.period):
            ss += 1
        ss -= 1
        print 'Subsampling by factor %d' % ss
        print 'Delta t effective %.2f' % (dt * ss)
        print 'No. samples per period %.1f' % (args.period / dt / ss)
        sampler = emcee.EnsembleSampler(args.nwalkers, ndim, lnprob_QP1, \
                                        args = (t[::ss], y[::ss], args.period))
        f = open(mcmc_output_file, 'w')
        f.close()
        for result in \
          sampler.sample(p0, iterations = args.nsteps, storechain = False):
            position = result[0]
            lnp = result[1]
            f = open(mcmc_output_file, 'a')
            for k in range(position.shape[0]):
                print ("{0:4d} {1:s}\n".format(k, " ".join(map(repr,position[k]))))
                f.write("{0:4d} {1:s}\n".format(k, " ".join(map(repr,position[k]))))
            f.close()
            raw_input('Take a look at %s' % mcmc_output_file)
    
    # labels = [r"$\log a$", r"$\log \tau_p$", \
    #           r"$P \, [\mathrm{days}]$", \
    #           r"$\log \tau_e \, [\mathrm{periods}]$", \
    #           r"$\log \sigma_w$"]
    # print("Running MCMC...")
    # sampler.run_mcmc(p0, 1000)
    
    # # Plot chains to check convergence
    # samples = plotchains(sampler, labels)
    # nwalker, nit, ndim = samples.shape
    # samples = samples.reshape((nwalker * nit, ndim))
    # pl.savefig('%s_QPGP_chains.png' % dfname)
    
    # # Plot posterior predictive mean and variance
    # ss = 1
    # while ((80 * dt * ss) < args.period):
    #     ss += 1
    # ss -= 1
    # nsamp_tpl = 10
    # pl.figure(1)
    # plotpred_QP1(s, t[::ss], y[::ss])
    # # for s in samples[np.random.randint(len(samples), size = nsamp_tpl)]:
    # #     plotsamples_QP1(s, t[::ss], y[::ss])
    # pl.xlim(t[0], t[-1])
    # pl.savefig('%s_QPGP_data.png' % dfname)

    # # Now do the corner plot
    # fig2 = corner.corner(samples, \
    #                      labels = labels, \
    #                      quantiles=[0.16, 0.5, 0.84], \
    #                      show_titles=True, title_kwargs={"fontsize": 12})
    # pl.savefig('%s_QPGP_corner_orig.png' % dfname)

    # # More intuitive parameters
    # samples_rescaled = samples.copy()
    # samples_rescaled[:,0] = np.exp(samples[:,0]) * 100
    # samples_rescaled[:,1] = np.exp(samples[:,1])
    # samples_rescaled[:,3] = np.exp(samples[:,3])
    # samples_rescaled[:,4] = np.exp(samples[:,4]) * 1e6
    # labels_rescaled = [r"$a \, [\%]$", r"$\tau_p$", \
    #                    r"$P \, [\mathrm{days}]$", \
    #                    r"$\tau_e \, [\mathrm{periods}]$", \
    #                    r"$\sigma_w \, [\mathrm{ppm}]$"]
    # fig4 = corner.corner(samples_rescaled, \
    #                      labels = labels_rescaled, \
    #                      quantiles=[0.16, 0.5, 0.84], \
    #                      show_titles=True, title_kwargs={"fontsize": 12})
    # pl.savefig('%s_QPGP_corner_rescaled.png' % dfname)
