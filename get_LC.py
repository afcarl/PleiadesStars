import numpy as np
import pylab as pl
from astropy.table import Table
from argparse import ArgumentParser
import subprocess

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('epic', metavar='E', type=int, help='EPIC number')
    args = ap.parse_args()

    # Download LC into current directory
    epic_str = '%09d' % args.epic
    epic_root = '%04s00000' % epic_str[:4]
    mast_root = 'https://archive.stsci.edu/missions/hlsp/k2sc/c04/%s' % \
      epic_root
    mast_file = 'hlsp_k2sc_k2_llc_%s-c04_kepler_v1_lc.fits' % epic_str
    subprocess.call(['wget', '%s/%s' % (mast_root, mast_file)])
        
    # Read it in
    t = Table.read(mast_file)
    mfl = t['mflags'].flatten()
    time = t['time'].flatten()
    flux_det = t['flux'].flatten()
    trend_t = t['trend_t'].flatten()
    err = t['error'].flatten()
    l = np.ones(len(time), 'bool')
    for bit in [3,4,6]:
        out = (mfl & (2**bit)) == (2**bit)
        l[out] = False
    l[np.isfinite(time) == False] = False
    l[np.isfinite(flux_det) == False] = False
    l[np.isfinite(trend_t) == False] = False
    l[np.isfinite(err) == False] = False
    m = np.median(trend_t[l])
    flux = flux_det + trend_t - m
    pl.clf()
    pl.plot(time, flux, 'r.')
    pl.plot(time[l], flux[l], 'k.')
    pl.show()

    # Save in ascii format
    ascii_file = 'epic%s.txt' % epic_str
    X = np.zeros((l.sum(),3))
    X[:,0] = time[l]
    X[:,1] = flux[l]
    X[:,2] = err[l]
    np.savetxt(ascii_file, X)
    
    # Remove FITS lc
    subprocess.call(['rm', '%s' % mast_file])
    
