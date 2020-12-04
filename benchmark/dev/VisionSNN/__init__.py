# -*- coding: utf-8 -*-
from .dot_input import generate_dot
from .probabilistic_2D_bump_attractor import Bump2D
from .postprocessing import PlotSNN

# changes with spiNNaker
# neuron model: if_cond_exp

import argparse
def init(args=[]):

    ###---CALLING ARGS----------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='2D bump attractor')

    parser.add_argument("--simulator", type=str, default="default", help="select nest or spyNNaker simulator")

    parser.add_argument("--tag", type=str, default='test', help="Tag for the experiment")
    parser.add_argument("--outpath", type=str, default='outputs', help="Folder to store data")
    parser.add_argument("--figpath", type=str, default='figures', help="Folder to store figures")

    parser.add_argument('--dt', type=float, default=1, help='time step for simulations (ms)')
    parser.add_argument('--min_delay', type=float, default=1, help='min delay (ms)')
    parser.add_argument('--max_delay', type=float, default=100, help='max delay (ms)')
    parser.add_argument('--simtime', type=float, default=1000, help='simulation time (ms)')

    parser.add_argument('--waveVelocity', type=float, default=0.1, help='wave velocity in mm/ms, delay = 1 + (0.001*d)*dpar, d=distance in um')
    parser.add_argument('--scalePar', type=float, default=1.0, help='distance scale factor (any distance scale can be used, provided it is used consistently.)')
    parser.add_argument('--E_weight', type=float, default=0.08, help='excitatory weight')
    parser.add_argument('--I_weight', type=float, default=0.08, help='inhibitory weight')
    parser.add_argument('--Eone_weight', type=float, default=0.08, help='excitatory weight one-to-one connections')
    parser.add_argument('--Ione_weight', type=float, default=0.08, help='inhibitory weight one-to-one connections')
    
    parser.add_argument('--E_expPar', type=float, default=1.0, help='par of the exp(-d**(1/par)) for exh synaspes')
    parser.add_argument('--I_expPar', type=float, default=2.0, help='par of the exp(-d**(1/par)) for inh synaspes')

    parser.add_argument('--E_v_tresh', type=float, default=-50, help='voltage treshold')
    parser.add_argument('--I_v_tresh', type=float, default=-50, help='voltage treshold')


    parser.add_argument('--N_pop', type=int, default=900, help='Population number in E and I layers')
    #parser.add_argument('--E_pop', type=float, default=900, help='number of E cells') #not used
    #parser.add_argument('--I_pop', type=float, default=900, help='number of I cells') #not used

    parser.add_argument('--im_noise', type=float, default=0.0, help='noise level')
    parser.add_argument('--im_contrast', type=float, default=0.1, help='contrast of the image')
    parser.add_argument('--N_tau', type=float, default=30., help='Characteristic time for computing the firing rate (ms)')
    parser.add_argument("--verbose", type=bool, default=True, help="Displays more verbose output.")
    parser.add_argument("--do_save", type=bool, default=True, help="Saves data in pickle files.")
    args = parser.parse_args(args=args)

    return args
