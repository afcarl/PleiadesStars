import numpy as np
from george import GP, kernels
import emcee, corner
import pylab as pl
from argparse import ArgumentParser
import os.path, subprocess
import matplotlib.gridspec as gridspec

# Syntax for converting to animated gif:
# convert -set delay 20 -set pause 200 -colorspace GRAY -colors 256 -dispose 1 -loop 0 -scale 50% epic210969800_QPGP_corner_???.png test.gif

def readLC(file):
    X = np.genfromtxt(file).T
    t, y, ye = X
    ym = y.mean()
    y = y / ym
    ye = ye / ym
    return t, y, ye

def lnlike_QP1(p, t, y):
    log_a, log_gamma, period, log_tau, log_sig_ppt = p
    a_sq, gamma, tau_sq, sig = \
      10.0**(2*log_a), 10.0**log_gamma, 10.0**(2*log_tau), \
      10.0**(log_sig_ppt-3)
    gp = GP(a_sq * kernels.ExpSine2Kernel(gamma, period) * \
            kernels.ExpSquaredKernel(tau_sq))
    yerr = np.ones(len(y)) * sig
    gp.compute(t, yerr)
    return gp.lnlikelihood(y)
    
def lnprior_QP1(p, per_init):
    """Jeffreys priors all round except for period where it's uniform"""
    log_a, log_gamma, period, log_tau, log_sig_ppt = p
    log_tau_rel = log_tau - np.log10(per_init) 
    if (log_a < -3) or (log_a > 0):
        return -np.inf
    if (log_gamma < -2.5) or (log_gamma > 1):
        return -np.inf
    if (period < (0.5 * per_init)) or (period > (2.0 * per_init)):
        return -np.inf
    if (log_tau_rel < 0.0) or (log_tau_rel > 2.0):
        return -np.inf    
    if (log_sig_ppt < -1) or (log_sig_ppt > 2):
        return -np.inf
    return 0.0

def lnprob_QP1(p, t, y, per_init):
    lp = lnprior_QP1(p, per_init)
    if np.isfinite(lp):
        ll = lnlike_QP1(p, t, y)
        return lp + ll
    else:
        return -np.inf

def plotchains(samples, lnp, labels, do_cut = False):
    nwalkers, nsteps, ndim = samples.shape
    fig = pl.figure(figsize=(8, ndim + 1))
    gs = gridspec.GridSpec(ndim+1, 1)
    gs.update(left=0.15, right=0.98, bottom = 0.1, top = 0.98, hspace=0.0)
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
    if do_cut == True:
        ans = int(raw_input('Enter number of iterations to discard as burn-in: '))
        for i in range(ndim+1):
            axc = pl.subplot(gs[i,0])
            pl.axvline(ans, color="#4682b4")
        print 'Discarding first %d steps as burn-in' % ans
        return fig, samples[:,ans:,:], lnp[:,ans:]
    else:
        return fig, samples, lnp
    
def plotsample_QP1(p, t, y):
    log_a, log_gamma, period, log_tau, log_sig_ppt = p
    a_sq, gamma, tau_sq, sig = \
      10.0**(2*log_a), 10.0**log_gamma, 10.0**(2*log_tau), \
      10.0**(log_sig_ppt-3)
    gp = GP(a_sq * kernels.ExpSine2Kernel(gamma, period) * \
            kernels.ExpSquaredKernel(tau_sq))
    yerr = np.ones(len(y)) * sig
    gp.compute(t, yerr)
    mu = gp.sample_conditional(y, t)
    pl.plot(t, mu, color="#4682b4", alpha=0.3)

def plotpred_QP1(p, t, y):
    log_a, log_gamma, period, log_tau, log_sig_ppt = p
    a_sq, gamma, tau_sq, sig = \
      10.0**(2*log_a), 10.0**log_gamma, 10.0**(2*log_tau), \
      10.0**(log_sig_ppt-3)
    gp = GP(a_sq * kernels.ExpSine2Kernel(gamma, period) * \
            kernels.ExpSquaredKernel(tau_sq))
    yerr = np.ones(len(y)) * sig
    gp.compute(t, yerr)
    mu, cov = gp.predict(y, t)
    sigma = np.diag(cov)
    sigma = np.sqrt(sigma**2 + yerr**2)
    pl.fill_between(t, mu + 2 * sigma, mu - 2 * sigma, color="#4682b4", alpha=0.3)
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
    ap.add_argument('--do-mcmc', action='store_true', default=False, \
                    help='re-run MCMC even if results file exists')
    ap.add_argument('--plot-progress', action='store_true', default=True, \
                    help='store incremental chain and corner plots after each batch of ~100 iterations')
    ap.add_argument('--plot-pred', action='store_true', default=True, \
                    help='in addition to chains and corner plot, plot LC with predictive distribution after MCMC has finished')
    args = ap.parse_args()

    t, y, ye = readLC(args.infile)
    dt = np.median(t[1:] - t[:-1])
    T = t[-1] - t[0]
    sig = (y[1:] - y[:-1]).std() / np.sqrt(2)
    ys = np.sort(y)
    n = len(y)
    h = ys[int(0.95*n)] - ys[int(0.05*n)]
    ye[:] = sig
    
    pl.figure(1)
    pl.errorbar(t, y, yerr = ye, fmt = 'k.', capsize = 0)
    pl.xlabel('time (d)')
    pl.ylabel('norm. flux')
    fname = os.path.splitext(os.path.split(args.infile)[-1])[0]
    dfname = '%s/%s' % (args.output_dir, fname)
    pl.title(fname)

    # do MCMC
    mcmc_output_file = '%s_QPGP_mcmc.npz' % dfname
    print 'Saving / looking for MCMC output in %s' % mcmc_output_file
    chainplot_root = '%s_QPGP_chains' % dfname
    cornerplot_root = '%s_QPGP_corner' % dfname
    labels = [r"$\log_{10}  \, a$", r"$\log_{10} \, \Gamma$", \
              r"$P \, [\mathrm{days}]$", \
              r"$\log_{10}  \, \tau [\mathrm{days}]$", \
              r"$\log_{10}  \, \sigma_w [ppt]$"]
    labels_nolatex = ['log_a', 'log_Gamma', 'P (d)', 'log_tau (d)', 'log_wn (ppt)']
    if (os.path.exists(mcmc_output_file) == False) or (args.do_mcmc == True):
        print 'Doing MCMC, %d walkers, %d iterations' % (args.nwalkers, args.nsteps)
        p_init = np.array([np.log10(np.sqrt(h)), -0.5, args.period, \
                           min(np.log10(args.period * 5.0), T), np.log10(sig)+3])
        ndim = len(p_init)
        p0 = [np.array(p_init) + 1e-8 * np.random.randn(ndim) \
            for i in xrange(args.nwalkers)]
        ss = 1
        while ((20 * dt * ss) < args.period):
            ss += 1
        ss -= 1
        print 'Subsampling by factor %d' % ss
        print 'Delta t effective %.2f' % (dt * ss)
        print 'No. samples per period %.1f' % (args.period / dt / ss)
        sampler = emcee.EnsembleSampler(args.nwalkers, ndim, lnprob_QP1, \
                                        args = (t[::ss], y[::ss], args.period))
        nsteps_batch = min(10, args.nsteps)
        nbatch = np.ceil(args.nsteps / nsteps_batch).astype('int')
        for i in range(nbatch):
            p0, _, _ = sampler.run_mcmc(p0, nsteps_batch)
            samples = sampler.chain
            lnp = sampler.lnprobability
            np.savez_compressed(mcmc_output_file, samples = samples, lnp = lnp)
            if args.plot_progress:                
                if i == nbatch - 1:
                    nsteps = min(nsteps_batch, args.nsteps - i * nsteps_batch)
                else:
                    nsteps = nsteps_batch
                if i > 0:
                    pl.close(fig2)
                    pl.close(fig3)
                fig2, samples, lnp = plotchains(samples, lnp, labels, do_cut = False)
                pl.savefig('%s_%03d.png' % (chainplot_root, i))
                nwalker, nit, ndim = samples.shape
                samples_flat = samples.reshape((nwalker * nit, ndim))
                fig3 = corner.corner(samples_flat, labels = labels, \
                                     quantiles=[0.16, 0.5, 0.84], \
                                     show_titles=True, title_kwargs={"fontsize": 12})
                pl.savefig('%s_%03d.png' % (cornerplot_root, i))
            print 'Batch %d of %d done (%d steps per batch)' % (i+1, nbatch, nsteps_batch)
        if args.plot_progress:
            pl.close(fig2)
            pl.close(fig3)
    else:
        data = np.load(mcmc_output_file)
        samples = data['samples']
        lnp = data['lnp']
        
    print 'Producing final plots'
    fig2, samples, lnp = plotchains(samples, lnp, labels, do_cut = True)
    nwalker, nit, ndim = samples.shape
    samples_flat = samples.reshape((nwalker * nit, ndim))
    pl.savefig('%s_final.png' % chainplot_root)    
    fig3 = corner.corner(samples_flat, \
                         labels = labels, \
                         quantiles=[0.16, 0.5, 0.84], \
                         show_titles=True, title_kwargs={"fontsize": 12})
    pl.savefig('%s_final.png' % cornerplot_root)

    nit = nwalker * nit
    print 'Total number of MCMC samples kept: %d' % nit
    imax = np.argmax(lnp.flatten())
    print '{0:20s} = {1:10s} + {2:10s} - {3:10s} ({4:10s})'.format('Parameter', 'median', '1sig', '1sig', 'MAP value')
    for i in range(len(labels_nolatex)):
        ss = samples_flat[:,i].flatten()
        most_prob = ss[imax]
        ss = np.sort(ss)
        med = ss[nit/2]
        top = ss[int(nit*0.84)]
        bot = ss[int(nit*0.16)]
        print '{0:20s} = {1:10g} + {2:10g} - {3:10g} ({4:10g})'.format(labels_nolatex[i], med, top-med, med-bot, most_prob)
        
    if args.plot_pred == True:
        print 'Plotting LC with predictive distribution'
        ss = 1
        while ((80 * dt * ss) < args.period):
            ss += 1
        ss -= 1
        nsamp_tpl = 10
        pl.figure(1)
        plotpred_QP1(samples_flat[imax,:], t[::ss], y[::ss])
        # for s in samples[np.random.randint(len(samples), size = nsamp_tpl)]:
        #     plotsamples_QP1(s, t[::ss], y[::ss])
        pl.xlim(t[0], t[-1])
        pl.savefig('%s_QPGP_pred.png' % dfname)
        pl.show()
