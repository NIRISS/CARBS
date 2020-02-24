import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, HTML

import corner
import emcee
import time
import sys
import copy

import pandas as pd
pd.set_option('display.max_colwidth', -1)

from grizli import utils, __version__
from grizli.aws import db as grizli_db
from grizli import multifit

try:
    from mastquery import overlaps
except:
    overlaps = None
    
#print('Grizli version: ', __version__)
utils.set_warnings()

import dense_basis as db
from tqdm import tqdm
import scipy.io as sio

from astropy.table import Table


#--------------------------------------------------------------------
#----------replace this with your own credentials -------------------
#--------------------------------------------------------------------

db_name = []
db_word = []
db_base = []

def import_AWS_credentials(fname):
    temp = open(fname,'r')
    db_name = temp.readline().strip()
    db_word = temp.readline().strip()
    db_base = temp.readline().strip()
    return db_name, db_word, db_base

#---------------------------------------------------------------------
#--------------- importing data from catalogs/AWS  -------------------
#---------------------------------------------------------------------


def import_phot_cat(phot_cat_path, z_cat_path):
    # load the goods-s 3D-HST catalog and try looking for matching objects
    # to-do: needs to be interfaced with Gabe's AWS database


    sed_cat = Table.read(phot_cat_path,
                        format = 'ascii.commented_header')

    redshift_cat = Table.read(z_cat_path,
                        format = 'ascii.commented_header')

    obs_id = sed_cat['id']
    obs_sed = sed_cat['f_F160W', 'f_U38','f_U','f_F435W','f_B','f_V',
                      'f_F606Wcand','f_F606W','f_R','f_Rc','f_F775W',
                      'f_I','f_F814Wcand','f_F850LP','f_F850LPcand',
                      'f_F125W','f_J','f_tenisJ','f_F140W','f_H','f_tenisK',
                      'f_Ks','f_IRAC1','f_IRAC2','f_IRAC3','f_IRAC4',
                      'f_IA427','f_IA445','f_IA505','f_IA527','f_IA550',
                      'f_IA574','f_IA598','f_IA624','f_IA651','f_IA679',
                      'f_IA738','f_IA767','f_IA797','f_IA856']

    obs_err = sed_cat['e_F160W', 'e_U38','e_U','e_F435W','e_B','e_V',
                      'e_F606Wcand','e_F606W','e_R','e_Rc','e_F775W',
                      'e_I','e_F814Wcand','e_F850LP','e_F850LPcand',
                      'e_F125W','e_J','e_tenisJ','e_F140W','e_H','e_tenisK',
                      'e_Ks','e_IRAC1','e_IRAC2','e_IRAC3','e_IRAC4',
                      'e_IA427','e_IA445','e_IA505','e_IA527','e_IA550',
                      'e_IA574','e_IA598','e_IA624','e_IA651','e_IA679',
                      'e_IA738','e_IA767','e_IA797','e_IA856']

    return obs_id, obs_sed, obs_err, sed_cat, redshift_cat


def load_grizli_database(db_name = db_name, db_word = db_word, db_base = db_base):
    """
    load AWS interface for getting stuff
    """

    # readonly DB access
    config = {'hostname': 'grizdbinstance.c3prl6czsxrm.us-east-1.rds.amazonaws.com',
     'username': db_name,
     'password': db_word,
     'database': db_base,
     'port': 5432}

    # sqlalchemy engine for postgresql
    engine = grizli_db.get_db_engine(config=config)
    #print('Tables: ', engine.table_names())
    
    return engine


def find_matched_obs(z_low = 1.6, z_high = 3.0, redshift_cat = [], sed_cat = [], engine = []):    
    
    cat_mask = (redshift_cat['z_spec'] > z_low) & (redshift_cat['z_spec'] < z_high) &\
    (np.abs(redshift_cat['z_spec'] - redshift_cat['z_a']) < 0.1) & (redshift_cat['nfilt'] > 39)
    print('# galaxies: %.0f' %np.sum(cat_mask))

    good_ids = redshift_cat['id'][cat_mask] - 1
    
    for i in range(len(good_ids)):
        gal_id = good_ids[i]
        sed_ra = sed_cat[gal_id]['ra']
        sed_dec = sed_cat[gal_id]['dec']
        sed_spec = redshift_cat[gal_id]['z_spec']
    #     print('gal_id: %.0f, z_spec: %.3f' %(gal_id+1, sed_spec))

        columns=['status','root','id','ra','dec','mag_auto','flux_radius', 'bic_diff', 'q_z','z_map','d4000','t_g102','t_g141']
        SQL = ("SELECT {colstr} FROM redshift_fit NATURAL JOIN photometry_apcorr" 
               " WHERE q_z > -0.2").format(colstr=','.join(columns))
        #        " AND z_map > 0.2 AND z_map < 0.3 AND mag_auto < 27").format(colstr=','.join(columns))
        extra = " AND ra < %.3f AND ra > %.3f" %(sed_ra + 0.001, sed_ra - 0.001)
        extra += " AND dec < %.3f AND dec > %.3f" %(sed_dec + 0.001, sed_dec - 0.001)
        extra += " AND z_map < %.3f AND z_map > %.3f" %(sed_spec + 0.005, sed_spec - 0.005)
        SQL = SQL + extra
        #print(SQL)

        res = grizli_db.from_sql(SQL, engine)
        #print('N: ', len(res))

        if (len(res) > 0):
            print('match for i = %.0f, with N: %.0f' %(i,len(res)))
            print('gal_id: %.0f, z_spec: %.3f' %(gal_id+1, sed_spec))
            so = np.argsort(res['mag_auto'])
#             HTML(grizli_db.render_for_notebook(res[so]['root','id','ra','dec','mag_auto','q_z','d4000','z_map'],image_extensions=['stack','full']))
    #     try:
    #         HTML(grizli_db.render_for_notebook(res[so]['root','id','ra','dec','mag_auto','q_z','d4000','z_map'],image_extensions=['stack','full']))
    #     except:
    #         print('i = %.0f, no matches' %i)
    
    return good_ids


def get_matched_multibeam(matched_id, good_ids, redshift_cat = [], sed_cat = [], engine = []):
        
    gal_id = good_ids[matched_id]
    sed_ra = sed_cat[gal_id]['ra']
    sed_dec = sed_cat[gal_id]['dec']
    sed_spec = redshift_cat[gal_id]['z_spec']
    print('gal_id: %.0f, z_spec: %.3f' %(gal_id+1, sed_spec))

    columns=['status','root','id','ra','dec','mag_auto','flux_radius', 'bic_diff', 'q_z','z_map','d4000','t_g102','t_g141']
    SQL = ("SELECT {colstr} FROM redshift_fit NATURAL JOIN photometry_apcorr" 
           " WHERE q_z > -0.2").format(colstr=','.join(columns))
    #        " AND z_map > 0.2 AND z_map < 0.3 AND mag_auto < 27").format(colstr=','.join(columns))
    extra = " AND ra < %.3f AND ra > %.3f" %(sed_ra + 0.001, sed_ra - 0.001)
    extra += " AND dec < %.3f AND dec > %.3f" %(sed_dec + 0.001, sed_dec - 0.001)
    extra += " AND z_map < %.3f AND z_map > %.3f" %(sed_spec + 0.005, sed_spec - 0.005)
    SQL = SQL + extra
    #print(SQL)

    res = grizli_db.from_sql(SQL, engine)
    print('N: ', len(res))

    so = np.argsort(res['mag_auto'])
    HTML(grizli_db.render_for_notebook(res[so]['root','id','ra','dec','mag_auto','q_z','d4000','z_map'],image_extensions=['stack','full']))

    match_obj = res[0]
    root, id = match_obj['root'].item(), match_obj['id'].item()
    print(root, id)

    # Fetch grism spectra file
    base_url = 'https://s3.amazonaws.com/grizli-v1/Pipeline/{0}/Extractions'.format(root)
    files = ['{0}_{1:05d}.beams.fits'.format(root, id), '{0}_fit_args.npy'.format(root)]
    for file in files: 
        #print(file)
        if not os.path.exists(file):
            os.system('wget {0}/{1}'.format(base_url, file))
            #print('wget {0}/{1}'.format(base_url, file))

    #args = np.load('{0}_fit_args.npy'.format(root), allow_pickle=True)[0]

    ix = (res['root'] == root) & (res['id'] == id)
    z_grism = res['z_map'][ix][0]
    print('Grism redshift: {0:.4f}'.format(z_grism))

    # let's load this spectrum in now:
    mb = multifit.MultiBeam('{0}_{1:05d}.beams.fits'.format(root, id))

    return mb, z_grism, gal_id


def get_matched_phot(mb, obs_sed, obs_err, gal_id, z_grism, filter_list = [], filt_dir = []):
    
    specstuff = mb.oned_spectrum()
    spec_lam = specstuff['G141']['wave']
    spec_flam = specstuff['G141']['flux']/specstuff['G141']['flat']
    spec_flam_err = specstuff['G141']['err']/specstuff['G141']['flat']
    
    filt_centers, filt_widths = filt_centers_rough(filter_list = filter_list, filt_dir = filt_dir, 
                                                   zval = z_grism, 
                                                   lam_arr = 10**np.linspace(2,8,10000), 
                                                   rest_frame = True, leff_method = 'median')
    
    phot_mask = ((filt_centers-filt_widths) > np.amin(spec_lam)) & ((filt_centers+filt_widths) < np.amax(spec_lam))
    
    temp = obs_sed.as_array()[gal_id]
    fitsed = np.array([i for i in temp])
    fitsed_flam = fitsed / ( 3.34e4 * (filt_centers**2) * 3 * 1e6)
    phot_fac = np.nanmedian(spec_flam)/np.nanmedian(fitsed_flam[phot_mask])
    fitsed_flam = fitsed_flam * phot_fac
    fitsed = fitsed * phot_fac
    temp = obs_err.as_array()[gal_id]
    fiterr = np.array([i for i in temp]) * phot_fac + (fitsed*0.03)
    fiterr_flam = fiterr / ( 3.34e4 * (filt_centers**2) * 3 *  1e6 ) 
    
    return fitsed_flam, fiterr_flam
    
#---------------------------------------------------------------------
#-------------------------interacting with FSPS-----------------------
#---------------------------------------------------------------------


def spec_from_FSPS(theta, stelmass, galmass):
    
    mstar, sfr, t25, t50, t75, Z, Av, z = theta
    sfh_tuple = np.array([mstar, sfr, 3.0, t25, t50, t75])
    
    db.mocksp.params['add_igm_absorption'] = True
    db.mocksp.params['zred'] = z
    db.mocksp.params['sfh'] = 3
    db.mocksp.params['cloudy_dust'] = True
    db.mocksp.params['dust_type'] = True
    
    db.mocksp.params['dust2'] = Av
    db.mocksp.params['logzsol'] = Z
    sfh, timeax = db.tuple_to_sfh(sfh_tuple, zval = z)
    db.mocksp.set_tabular_sfh(timeax, sfh)
    wave, spec = db.mocksp.get_spectrum(tage = np.amax(timeax), peraa = True)
    
    stelmass.append(np.log10(db.mocksp.stellar_mass))
    galmass.append(mstar)
    
    return spec, wave, stelmass, galmass


def sed_from_FSPS(theta, stelmass, galmass, fcs, zgrid):
    
    spec, wave, stelmass, galmass = spec_from_FSPS(theta, stelmass, galmass,)
    
    zarg = np.argmin(np.abs(zgrid - theta[-1]))
    filcurves = fcs[0:,0:,zarg]
    sed = db.calc_fnu_sed_fast(spec, filcurves)
    
    return sed, stelmass, galmass


def convert_sed_to_flam(sed, sederr, filt_centers):
    
    sed_flam = sed / ( 3.34e4 * (filt_centers**2) * 3 * 1e6)
    sed_err_flam = sederr / ( 3.34e4 * (filt_centers**2) * 3 * 1e6)
    
    return sed_flam, sed_err_flam
    
    
def get_spec_from_mb(mb):
    
    specstuff = mb.oned_spectrum()

    spec_lam = specstuff['G141']['wave']
    spec_flam = specstuff['G141']['flux']/specstuff['G141']['flat']
    spec_flam_err = specstuff['G141']['err']/specstuff['G141']['flat']
    
    return spec_lam, spec_flam, spec_flam_err

    
def phot_scale_factor(sed, filt_centers, filt_widths, mb):
    
    spec_lam, spec_flam, spec_flam_err = get_spec_from_mb(mb)
    
    phot_mask = ((filt_centers-filt_widths) > np.amin(spec_lam)) &\
        ((filt_centers+filt_widths) < np.amax(spec_lam))
    
    phot_fac = np.nanmedian(spec_flam)/np.nanmedian(sed[phot_mask])
    
    return phot_fac


def spec_norm(z_grism):
    
    # FSPS outputs Lsun/AA
    # spec * lsun = spec in ergs/s/AA
    # spec * lsun / (4 pi dL^2) # get flux from luminosity by dividing by surface area
    
    dlfac = 1 / (4* np.pi * db.cosmo.luminosity_distance(z_grism).to('cm').value**2)
    lsun = 3.846e33 # ergs/s
    corrfac = (lsun)*(dlfac)
    return corrfac


#---------------------------------------------------------------------
#--------------------filter transmission utilities--------------------
#---------------------------------------------------------------------

def filt_centers_rough(filter_list = 'filter_list_goodss.dat', filt_dir = 'filters/', zval = 1.0, lam_arr = 10**np.linspace(2,8,10000), rest_frame = True, leff_method = 'median'):
    
    filcurves, lam_z, lam_z_lores = db.make_filvalkit_simple(lam_arr, zval, fkit_name = filter_list, filt_dir = filt_dir)
    filt_centers = np.zeros((filcurves.shape[1]))
    filt_widths = np.zeros((filcurves.shape[1]))
    for i in range(len(filt_centers)):
        if leff_method == 'max':
            filt_centers[i] = lam_arr[np.argmax(filcurves[0:,i])]*(1+zval)
        elif leff_method == 'median':
            med_index = np.argmin(np.abs(np.cumsum(filcurves[0:,i])/np.amax(np.cumsum(filcurves[0:,i])) - 0.5))
            lo_index = np.argmin(np.abs(np.cumsum(filcurves[0:,i])/np.amax(np.cumsum(filcurves[0:,i])) - 0.16))
            hi_index = np.argmin(np.abs(np.cumsum(filcurves[0:,i])/np.amax(np.cumsum(filcurves[0:,i])) - 0.84))
            filt_centers[i] = lam_arr[med_index]*(1+zval)
            filt_widths[i] = lam_arr[hi_index]*(1+zval) - lam_arr[lo_index]*(1+zval)
        else:
            print('unknown leff_method: use max or median')
    return filt_centers, filt_widths

def make_fcs(lam, z_min, z_max, z_step = 0.01, filter_list = [], filt_dir = []):

    fc_zgrid = np.arange(z_min-z_step, z_max+2*z_step, z_step)

    temp_fc, temp_lz, temp_lz_lores = db.make_filvalkit_simple(lam, z_min,fkit_name = filter_list, filt_dir = filt_dir)

    fcs = np.zeros((temp_fc.shape[0], temp_fc.shape[1], len(fc_zgrid)))
    lzs = np.zeros((temp_lz.shape[0], len(fc_zgrid)))
    lzs_lores = np.zeros((temp_lz_lores.shape[0], len(fc_zgrid)))

    for i in tqdm(range(len(fc_zgrid))):
        fcs[0:,0:,i], lzs[0:,i], lzs_lores[0:,i] = db.make_filvalkit_simple(lam,fc_zgrid[i],fkit_name = filter_list, filt_dir = filt_dir)

    return fcs, fc_zgrid

def get_pg_theta(z_grism, filter_list, filt_dir):
    
    # initialize a priors object
    priors = db.Priors()
    priors.z_min = z_grism-1e-3
    priors.z_max = z_grism+1e-3
    priors.Av_min = 0.0
    #priors.Av_max = args['MW_EBV']*10
    priors.Av_max = 1

    filt_centers, filt_widths = filt_centers_rough(filter_list, filt_dir, leff_method = 'median')

    fname = 'test_atlas'
    N_pregrid = 100
    priors.Nparam = 3
    db.generate_atlas(N_pregrid = N_pregrid,
                      priors = priors,
                      fname = fname, store=True,
                      filter_list = filter_list, filt_dir = filt_dir)

    path = 'pregrids/'
    
    pg_sfhs, pg_Z, pg_Av, pg_z, pg_seds, norm_method = db.load_atlas(fname, N_pregrid = N_pregrid, N_param = priors.Nparam, path = path)
    pg_params = np.vstack([pg_sfhs[0,0:], pg_sfhs[1,0:], pg_sfhs[3:,0:], pg_Z, pg_Av, pg_z])
    
    return pg_params

#---------------------------------------------------------------------
#------------------ emcee likelihood functions -----------------------
#---------------------------------------------------------------------

def lnprior(theta, z_grism, txpad = 0.05):
    # priors for the sampler, set this up to import from the db.Priors() object
    # and also generalize to any number of tx parameters
    
    mstar, sfr, t25, t50, t75, Z, Av, z = theta
    if 9.0 < mstar < 12.0 and -3.0 < sfr < 3.0 and\
    (0.1+txpad) < t25 < (t50-txpad) and\
    (t25+txpad) < t50 < (t75-txpad) and\
    (t50+txpad) < t75 < (1.0-txpad) and\
    -1.5 < Z < 0.5 and 0.0 < Av < 1.0 and\
    (z_grism-1e-3)<z<(z_grism+1e-3):
        return 0.0
    return -np.inf


# likelihood chi^2
def lnlike_grism(theta, mb, stelmass, galmass):
    
    spec, wave, stelmass, galmass = spec_from_FSPS(theta, stelmass, galmass)
    spec_scaled = spec * spec_norm(theta[-1])
    
    templ = {'fsps':utils.SpectrumTemplate(wave=wave, flux=spec_scaled, name='fsps')}
    tfit = mb.template_at_z(theta[-1], templates=templ)
    chi2 = tfit['chi2']/mb.Nspec
    
    return np.sum(-chi2/2)


def lnprob_grism(theta, mb, stelmass, galmass, z_grism):
    lp = lnprior(theta, z_grism)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_grism(theta, mb, stelmass, galmass)


def lnlike_phot(theta, sedobs, sederr, fcs, zgrid, stelmass, galmass):
    
    model_sed, stelmass, galmass = sed_from_FSPS(theta, stelmass, galmass, fcs, zgrid)
    model_sed = model_sed * spec_norm(theta[-1])
    fit_mask = (sedobs > 0) #& (~np.isnan(sed))
    chi2 = np.sum(((model_sed - sedobs)**2)/((sederr)**2)) / np.sum(fit_mask)
    return np.sum(-chi2/2)


def lnprob_phot(theta, sedobs, sederr, fcs, zgrid, stelmass, galmass, z_grism):
    lp = lnprior(theta, z_grism)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_phot(theta, sedobs, sederr, fcs, zgrid, stelmass, galmass)


def lnlike_grismphot(theta, mb, sedobs, sederr, fcs, zgrid, stelmass, galmass, chi2wt, printchi2 = False):
    
    spec, wave, stelmass, galmass = spec_from_FSPS(theta, stelmass, galmass)
    spec = spec * spec_norm(theta[-1])
    
    templ = {'fsps':utils.SpectrumTemplate(wave=wave, flux=spec, name='fsps')}
    tfit = mb.template_at_z(theta[-1], templates=templ)
    chi2_spec = tfit['chi2']/mb.Nspec
    
    zarg = np.argmin(np.abs(zgrid - theta[-1]))
    filcurves = fcs[0:,0:,zarg]
    model_sed = db.calc_fnu_sed_fast(spec, filcurves)
    fit_mask = (sedobs > 0) #& (~np.isnan(sed))
    chi2_phot = np.sum(((model_sed - sedobs)**2)/((sederr)**2)) / np.sum(fit_mask)
    
    chi2 = (chi2_spec * chi2wt) + (chi2_phot * (1-chi2wt))
    if printchi2 == True:
        print('chi2/DoF from grism: %.2f and phot: %.2f.'%((chi2_spec) , (chi2_phot)))
    return np.sum(-chi2/2)
    
    
def lnprob_grismphot(theta, mb, sedobs, sederr, fcs, zgrid, stelmass, galmass, chi2wt, z_grism):
    lp = lnprior(theta, z_grism)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_grismphot(theta, mb, sedobs, sederr, fcs, zgrid, stelmass, galmass, chi2wt)


#---------------------------------------------------------------------
#------------------ actual emcee wrapper functions -------------------
#---------------------------------------------------------------------


def db_mcmc(mb = [], sedobs = [], sederr = [], fcs = [], zgrid = [], z_grism = [], pg_params = [], fit_type='grismphot', n_walkers = 100, n_steps = 100, n_burnin = 10, threads = 6, chi2wt = 0.5, printf=False, return_sampler = False, filter_list = [], filt_dir = []):
    """
    Run emcee on spectrophotometry to determine optimal physical properties from galaxy SEDs.
    
    to-do: write unit-tests
    """
    
    ndim = pg_params.shape[0]
    
    stelmass = []
    galmass = []

    pos = pg_params[0:,np.random.choice(pg_params.shape[1], size = n_walkers)].T
    pos[0:, 0] = np.random.random(size=n_walkers)*3.0 + 9.0
    pos[0:, 1] = np.random.random(size=n_walkers)*6.0 - 3.0
    pos[0:,-1] = z_grism
    
    if fit_type == 'grism':
        sampler = emcee.EnsembleSampler(n_walkers, ndim, 
                                        lnprob_grism, 
                                        args = (mb, stelmass, galmass, z_grism), 
                                        threads = threads)
    elif fit_type == 'phot':
        
        sampler = emcee.EnsembleSampler(n_walkers, ndim, 
                                        lnprob_phot, 
                                        args = (sedobs, sederr, fcs, zgrid, stelmass, galmass, z_grism), 
                                        threads = threads)
    elif fit_type == 'grismphot':
        
        filt_centers, filt_widths = filt_centers_rough(filter_list = filter_list, filt_dir = filt_dir, 
                                                   zval = z_grism, 
                                                   lam_arr = 10**np.linspace(2,8,10000), 
                                                   rest_frame = True, leff_method = 'median')
        
        pfac = phot_scale_factor(sedobs, filt_centers, filt_widths, mb)
        # print('mean offset between grism and photometry: %.3f' %pfac)
        sampler = emcee.EnsembleSampler(n_walkers, ndim, 
                                        lnprob_grismphot, 
                                        args = (mb, sedobs*pfac, sederr*pfac, fcs, zgrid, stelmass, galmass, chi2wt, z_grism), 
                                        threads = threads)
        
    time0 = time.time()
    pos, prob, state = sampler.run_mcmc(pos, n_burnin)
    sampler.reset()
    time1 = time.time()
    print('burn-in time: %.1f sec' %(time1-time0))

    time0 = time.time()
    
    width = 100
    for i, result in enumerate(sampler.sample(pos, iterations = n_steps)):
        n = int((width+1)*float(i)/n_steps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n , ' '*(width-n)))
    sys.stdout.write("\n")

    time1 = time.time()
    print('time taken to run: %.1f min.' %((time1-time0)/60))

    samples = sampler.flatchain
    
    #------------------------- fits are done, do renorm -----------------------------
    
    # median parameter estimate
    bf_theta = np.nanmedian(samples,0)
    if fit_type == 'grismphot':
        chi2 = lnlike_grismphot(bf_theta, mb, sedobs*pfac, sederr*pfac, fcs, zgrid, stelmass, galmass, chi2wt, printchi2 = True)
    spec, wave, stelmass, galmass = spec_from_FSPS(bf_theta, stelmass, galmass)
    bf_spec_flam = spec * spec_norm(z_grism)
    
    specstuff = mb.oned_spectrum()
    spec_lam = specstuff['G141']['wave']
    spec_flam = specstuff['G141']['flux']/specstuff['G141']['flat']
    spec_flam_err = specstuff['G141']['err']/specstuff['G141']['flat']
    
    wave_mask = (wave > np.amin(spec_lam)/(1+z_grism)) & (wave < np.amax(spec_lam)/(1+z_grism))
    corrfac = np.nanmedian(spec_flam) / np.nanmedian(bf_spec_flam[wave_mask])
    print('overall offset between median spec and grism: %.3f' %corrfac)
    
#     samples[0:,0] = samples[0:,0] + np.log10(corrfac) - np.nanmedian(np.array(galmass) - np.array(stelmass))
#     samples[0:,1] = samples[0:,1] + np.log10(corrfac)
    
#     samples[0:,0] = samples[0:,0] - np.nanmedian(np.array(galmass) - np.array(stelmass))
#     samples[0:,1] = samples[0:,1]
    if return_sampler == True:
        return sampler
    else:
        return samples, stelmass, galmass


#---------------------------------------------------------------------
#------------------------- posterior plots ---------------------------
#---------------------------------------------------------------------

def plot_matched_spectrophotometry(mb, z_grism, gal_id, obs_sed, obs_err, filter_list, filt_dir):
    
    specstuff = mb.oned_spectrum()
    spec_lam = specstuff['G141']['wave']
    spec_flam = specstuff['G141']['flux']/specstuff['G141']['flat']
    spec_flam_err = specstuff['G141']['err']/specstuff['G141']['flat']
    
    filt_centers, filt_widths = filt_centers_rough(filter_list = filter_list, filt_dir = filt_dir, 
                                                   zval = z_grism, 
                                                   lam_arr = 10**np.linspace(2,8,10000), 
                                                   rest_frame = True, leff_method = 'median')
    
    phot_mask = ((filt_centers-filt_widths) > np.amin(spec_lam)) & ((filt_centers+filt_widths) < np.amax(spec_lam))
    
    temp = obs_sed.as_array()[gal_id]
    fitsed = np.array([i for i in temp])
    fitsed_flam = fitsed / ( 3.34e4 * (filt_centers**2) * 3 * 1e6)
    phot_fac = np.nanmedian(spec_flam)/np.nanmedian(fitsed_flam[phot_mask])
    fitsed_flam = fitsed_flam * phot_fac
    fitsed = fitsed * phot_fac
    temp = obs_err.as_array()[gal_id]
    fiterr = np.array([i for i in temp]) * phot_fac + (fitsed*0.03)
    fiterr_flam = fiterr / ( 3.34e4 * (filt_centers**2) * 3 *  1e6 ) 
    
    
    plt.figure(figsize=(12,6))
    
    
    
    plt.errorbar(filt_centers/1e4, fitsed_flam,yerr = fiterr_flam, xerr=filt_widths/2/1e4,marker='o',lw=0,elinewidth=2,capsize=5)
    plt.errorbar(filt_centers[phot_mask]/1e4, fitsed_flam[phot_mask],
                 yerr = fiterr_flam[phot_mask], xerr = filt_widths[phot_mask]/2/1e4,marker='o',lw=0,elinewidth=2,capsize=5)
    
    plt.errorbar(spec_lam/1e4, spec_flam, spec_flam_err,lw=0,elinewidth=2,marker='.',alpha=0.3)
    plt.xscale('log')
    # plt.ylim(0,np.amax(fitsed)*1.2)
    plt.xlabel('$\lambda$ [micron]')
    plt.ylabel(r'$F_\lambda$ [ergs/(cm$^2$s$\AA$)]')
    #plt.axis([0.3,1,0,20000])
    #plt.axis([1.1,1.7,0,np.nanmax(spec_flam)*1.2])
    plt.show()

    
def plot_all_fits(samples, stelmass, galmass, mb, z_grism, fcs, fc_zgrid, temp_sed, temp_err, filter_list, filt_dir, num_sfh = 1000, scaleto = 'phot'):
    
#     temp = obs_sed.as_array()[gal_id]
#     fitsed = np.array([i for i in temp])
#     temp = obs_err.as_array()[gal_id]
#     fiterr = np.array([i for i in temp])
#     temp_sed, temp_err = convert_sed_to_flam(fitsed, fiterr, filt_centers)
    
    filt_centers, filt_widths = filt_centers_rough(filter_list = filter_list, filt_dir = filt_dir, 
                                                   zval = z_grism, 
                                                   lam_arr = 10**np.linspace(2,8,10000), 
                                                   rest_frame = True, leff_method = 'median')

    
    
    spec_lam, spec_flam, spec_flam_err = get_spec_from_mb(mb)

    pfac = phot_scale_factor(temp_sed, filt_centers, filt_widths, mb)
    temp_sed = temp_sed * pfac
    temp_err = temp_err * pfac
#     print(pfac)

    #temp_theta = np.array([10.7,-1.0,0.25,0.5,0.75,0.0,0.0,z_grism])
    temp_theta = np.median(samples,0)
    temp_theta[0] = temp_theta[0] #+ np.log10(1)
    temp_theta[1] = temp_theta[1] #+ np.log10(1)

    spec, wave, stelmass, galmass = spec_from_FSPS(temp_theta, stelmass, galmass)
    specfac = spec_norm(z_grism)
    spec = spec * specfac

    templ = {'fsps':utils.SpectrumTemplate(wave=wave, flux=spec, name='fsps')}
    tfit = mb.template_at_z(z_grism, templates=templ)

    # print(specfac)
    zarg = np.argmin(np.abs(fc_zgrid - z_grism))
    filcurves = fcs[0:,0:,zarg]
    model_sed = db.calc_fnu_sed_fast(spec, filcurves)

    wave_mask = (wave > np.amin(spec_lam)/(1+z_grism)) & (wave < np.amax(spec_lam)/(1+z_grism))
    if scaleto == 'phot':
        corrfac = np.nanmedian(temp_sed[temp_sed>0])/np.nanmedian(model_sed[temp_sed>0])
    elif scaleto == 'grism':
        corrfac = np.nanmedian(spec_flam) / np.nanmedian(spec[wave_mask])
        print('overall offset between median spec and grism: %.3f' %corrfac)
    else:
        corrfac = 1.0
        print('unscaled SED, can not trust mass/sfr estimates')
    
    plt.figure(figsize=(12,6))
    plt.errorbar(filt_centers, temp_sed*1e19, xerr = filt_widths/2,yerr = temp_err/2*1e19 ,marker='o',label='obs_phot', lw=0, elinewidth=2,capsize=5)
    plt.errorbar(spec_lam, spec_flam*1e19, yerr = spec_flam_err*1e19,label='obs_grism')
    plt.plot(wave*(1+z_grism), spec*corrfac*1e19,'k', alpha=0.3,label='median_model_spec')
    plt.plot(filt_centers, model_sed*corrfac*1e19,'ko', alpha=0.3,label='median_model_phot')

    plt.xlim(3e3,1e5)
    plt.ylim(0,np.amax(temp_sed)*1.2*1e19)
    plt.xscale('log')
    plt.xlabel('$\lambda$ [$\AA$]')
    plt.ylabel(r'F$_\lambda \times 10^{19}$')
    plt.legend(edgecolor='w')
    plt.show()

    fig = mb.oned_figure(tfit=tfit, figsize=(12,6))
    plt.show()

    plot_emcee_posterior(samples[0:,0:7], stelmass, galmass, corrfac)

    fig = plot_sfh_posteriors(samples, num_sfh = num_sfh)
    plt.show()
    
    return

def plot_sfh_posteriors(unnormed_samples, stelmass = [], galmass = [], num_sfh = 1000, corrfac = 1.0):
    
    samples = unnormed_samples.copy()
    samples[0:,0] = samples[0:,0] + np.log10(corrfac)
    samples[0:,1] = samples[0:,1] + np.log10(corrfac)
    
#     median_stelmass = np.nanmedian(np.array(stelmass))
#     median_galmass = np.nanmedian(np.array(galmass))
#     samples[0:,0] = samples[0:,0] + np.log10(median_stelmass) - np.log10(median_galmass) + np.log10(corrfac)
#     samples[0:,1] = samples[0:,1] + np.log10(median_stelmass) - np.log10(median_galmass) + np.log10(corrfac)
    
    sfhs = np.zeros((1000, num_sfh))
    for i in tqdm(range(num_sfh)):
        
        mstar = samples[-(i+1),0]
        sfr = samples[-(i+1),1]
        t25 = samples[-(i+1),2]
        t50 = samples[-(i+1),3]
        t75 = samples[-(i+1),4]
        sfh_tuple = np.array([mstar, sfr, 3.0, t25, t50, t75])
        sfhs[0:,i], timeax = db.tuple_to_sfh(sfh_tuple, samples[-(i+1),-1])
    
    fig = plt.figure(figsize=(12,6))
    plt.plot(np.amax(timeax) - timeax, np.nanmedian(sfhs,1), lw=3)
    plt.fill_between(np.amax(timeax) - timeax, 
                     np.nanpercentile(sfhs,16,1),np.nanpercentile(sfhs,84,1),
                     alpha=0.1)
    plt.ylabel('SFR(t) [M$_\odot.yr^{-1}$]')
    plt.xlabel('t [lookback time; Gyr]')
    plt.ylim(0,np.amax(np.nanmedian(sfhs,1))*1.5)
    #plt.show()
    return fig


def plot_emcee_posterior(unnormed_samples, stelmass, galmass, corrfac = 1.0, sed_truths = []):

    samples = unnormed_samples.copy()
    median_stelmass = np.nanmedian(np.array(stelmass))
    median_galmass = np.nanmedian(np.array(galmass))
    samples[0:,0] = samples[0:,0] + np.log10(median_stelmass) - np.log10(median_galmass) + np.log10(corrfac)
    samples[0:,1] = samples[0:,1] + np.log10(median_stelmass) - np.log10(median_galmass) + np.log10(corrfac)
    
    if len(sed_truths) > 1:
        fig = corner.corner(samples, labels = ['log M*', 'log SFR', 't$_{25}$', 't$_{50}$', 't$_{75}$', 'log Z/Z$_\odot$', 'A$_V$', 'redshift'],
                            truths = sed_truths,
                            plot_datapoints=False, fill_contours=True,
                            bins=20, smooth=1.0,
                            quantiles=(0.16, 0.84), levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                            label_kwargs={"fontsize": 30}, show_titles=True)

    else:
        fig = corner.corner(samples, labels = ['log M*', 'log SFR', 't$_{25}$', 't$_{50}$', 't$_{75}$', 'log Z/Z$_\odot$', 'A$_V$', 'redshift'],
                            plot_datapoints=False, fill_contours=True,
                            bins=20, smooth=1.0,
                            quantiles=(0.16, 0.84), levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                            label_kwargs={"fontsize": 30}, show_titles=True)

    fig.subplots_adjust(right=1.5,top=1.5)
    #fig.set_size_inches(12,12)
    plt.show() 
    
def plot_spec_fit(samples, z_grism, mb):

    bf_theta = np.nanmedian(samples,0)
    spec, wave, stelmass = spec_from_FSPS(bf_theta)
    templ = {'fsps':utils.SpectrumTemplate(wave=wave, flux=spec, name='fsps')}
    tfit = mb.template_at_z(z_grism, templates=templ)
    fig = mb.oned_figure(tfit=tfit, figsize=(12,6))
    
    return fig