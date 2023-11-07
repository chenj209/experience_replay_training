import numpy as np
import torch

def torch_get_pmid(ps, hyam, hybm):
    # Use unsqueeze to add an extra dimension to ps, equivalent to np.newaxis
    pmid = ps.unsqueeze(1) * hybm.unsqueeze(0)
    pmid += hyam.unsqueeze(0) * 100000
    
    return pmid

def torch_cal_qsat_water(t, p):
    # t: temperature (30, 96, 144)
    # p: pressure (30,96, 144)
    # Constants
    ps = torch.tensor([1013.246], device='cuda')
    ts = torch.tensor([373.16], device='cuda')
    e1 = 11.344 * (1.0 - t / ts)
    e2 = -3.49149 * (ts / t - 1.0)
    f1 = -7.90298 * (ts / t - 1.0)
    f2 = 5.02808 * torch.log10(ts / t)
    f3 = -1.3816 * (10.0 ** e1 - 1.0) / 10000000.0
    f4 = 8.1328 * (10.0 ** e2 - 1.0) / 1000.0
    f5 = torch.log10(ps)
    f = f1 + f2 + f3 + f4 + f5
    es = (10.0 ** f) * 100.0
    epsqs = torch.tensor([0.622], device='cuda')  # Assuming epsqs is a constant with a value of 0.622
    qsat_water = epsqs * es / (p - (1.0 - epsqs) * es)  # saturation w/respect to liquid only
    qsat_water = torch.where(qsat_water < 0.0, torch.tensor([1.0], device='cuda', dtype=qsat_water.dtype), qsat_water)
    return qsat_water


def get_pmid_from_x(x, hyam, hybm):
    ps = x[:, 121, :, :]
    pmid = ps[:, np.newaxis, :, :] * np.array(hybm)[np.newaxis, :, np.newaxis, np.newaxis]
    pmid += np.array(hyam)[np.newaxis, :, np.newaxis, np.newaxis] * 100000
    return pmid

def get_pmid_from_ps1d(ps, hyam, hybm):
    """
    ps: (N, 1)
    """
    pmid = ps[:, np.newaxis] * np.array(hybm)[np.newaxis, :]
    pmid += np.array(hyam)[np.newaxis, :] * 100000
    return pmid


def qsat_water_single(t, p):
    # Constants
    ps = 1013.246
    ts = 373.16
    e1 = 11.344 * (1.0 - t / ts)
    e2 = -3.49149 * (ts / t - 1.0)
    f1 = -7.90298 * (ts / t - 1.0)
    f2 = 5.02808 * np.log10(ts / t)
    f3 = -1.3816 * (10.0 ** e1 - 1.0) / 10000000.0
    f4 = 8.1328 * (10.0 ** e2 - 1.0) / 1000.0
    f5 = np.log10(ps)
    f = f1 + f2 + f3 + f4 + f5
    es = (10.0 ** f) * 100.0
    
    epsqs = 0.622  # Assuming epsqs is a constant with a value of 0.622
    qsat_water = epsqs * es / (p - (1.0 - epsqs) * es)  # saturation w/respect to liquid only
    if (qsat_water<0):
        qsat_water = 1
    
    return qsat_water

def cal_rh_single(q,t,p):
    return q/qsat_water_single(t,p)

def qsat_water(t, p):
    # t: temperature (30, 96, 144)
    # p: pressure (30,96, 144)

    # Convert input arrays to NumPy arrays and broadcast p to match t's shape
    t = np.asarray(t)
    p = np.asarray(p)
#     p = np.asarray(p)[np.newaxis, :, :]

    # Constants
    ps = 1013.246
    ts = 373.16
    e1 = 11.344 * (1.0 - t / ts)
    e2 = -3.49149 * (ts / t - 1.0)
    f1 = -7.90298 * (ts / t - 1.0)
    f2 = 5.02808 * np.log10(ts / t)
    f3 = -1.3816 * (10.0 ** e1 - 1.0) / 10000000.0
    f4 = 8.1328 * (10.0 ** e2 - 1.0) / 1000.0
    f5 = np.log10(ps)
    f = f1 + f2 + f3 + f4 + f5
    es = (10.0 ** f) * 100.0
    
    epsqs = 0.622  # Assuming epsqs is a constant with a value of 0.622
    qsat_water = epsqs * es / (p - (1.0 - epsqs) * es)  # saturation w/respect to liquid only
    qsat_water[qsat_water < 0.0] = 1.0
    
    return qsat_water
def cal_rh(q, t, p):
    return q/qsat_water(t,p)
