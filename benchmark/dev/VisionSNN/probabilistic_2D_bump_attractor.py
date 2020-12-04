###
# updated version of the file 2020-08-06
#
# code for running the simulation of a 2D bump attractor network
#
# it loads the flash lag 2D + T input and generates spike source correctly

###---LIBRARIES
import os
import numpy as np
import pickle
from .dot_input import generate_dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Bump2D():
    def __init__(self, args):
        self.args = args
        self.d = vars(args)

        os.makedirs(self.args.figpath, exist_ok=True)

        if self.args.do_save:
            self.tagpath = os.path.join(self.args.outpath, self.args.tag)
            self.figpath = os.path.join(self.tagpath, self.args.figpath)
            os.makedirs(self.figpath + '/moving_bump/E_layer/', exist_ok=True)
            os.makedirs(self.figpath + '/moving_bump/I_layer/', exist_ok=True)


    def run(self):
        try:
            self.popEdata = pickle.load(open(os.path.join(self.tagpath, 'popEdata.pkl'), 'rb'))
            self.popIdata = pickle.load(open(os.path.join(self.tagpath, 'popIdata.pkl'), 'rb'))
            self.spikeSources = pickle.load(open(os.path.join(self.tagpath, 'spikeSource.pkl'),'rb'))
            self.projEE = np.loadtxt(os.path.join(self.tagpath, 'weights_projEE.txt'), dtype='float')
            #self.projEI = np.loadtxt(os.path.join(self.tagpath, 'weights_projEI.txt'), dtype='float')
            self.projEIone = np.loadtxt(os.path.join(self.tagpath, 'weights_projEIone.txt'), dtype='float')
            self.projII = np.loadtxt(os.path.join(self.tagpath, 'weights_projII.txt'), dtype='float')
            #self.projIE = np.loadtxt(os.path.join(self.tagpath, 'weights_projIE.txt'), dtype='float')
            self.projIEone = np.loadtxt(os.path.join(self.tagpath, 'weights_projIEone.txt'), dtype='float')

            print('Loaded data')

        except:
            from pyNN.space import Grid2D
            if self.args.simulator == 'default':
                try:
                    import pyNN.spiNNaker as sim
                except ModuleNotFoundError:
                    import pyNN.nest as sim
            else:
                import importlib
                sim = importlib.import_module(self.args.simulator)


            ###---LOADING INPUT -------------------------------------------------------------------------------
            self.time_bins = int(self.args.simtime/self.args.dt) #extended time
            #extended pixels
            width = int(np.sqrt(self.args.N_pop))
            #1.1 generate 2d with time stimulus
            inputBar = generate_dot(width, width, self.time_bins,
                                    dot_size=0.1,
                                    flash_start=0.2,
                                    flash_duration=0.6,
                                    im_noise=self.args.im_noise,
                                    im_contrast=self.args.im_contrast,
                                    )
            inputBar = inputBar.transpose(2, 1, 0)
            randomness =  np.random.rand(self.time_bins, width, width)
            #1.2 thresholding
            inputBarNR = inputBar > randomness #inputBar Not Random
            inputBarNR[:2, :] = 0 # avoid having a spike at the first time bin

            #cellSourcesID = []
            #for i in range(time_bins):
            #    cellSourcesID.append(np.where(inputBarNR[i].reshape(self.args.N_pop,1)==True))

            cellSourceSpikes = []
            for i in range(self.args.N_pop):
                row = i//width
                col = i%width
                spike_times = np.where(inputBarNR[:, row, col]==True)[0]
                cellSourceSpikes.append(spike_times)

            cellSourceSpikes = [list(elem) for elem in cellSourceSpikes]

            ###---SIMULATION SETUP-----------------------------------------------------------------------------
            sim.setup(timestep=self.args.dt,
                min_delay=self.args.min_delay,
                max_delay=self.args.max_delay, #with Nest 1000
                debug=0)

            ###---CELL PARAMETERS-------------------------------------------------------------------------------
            cell_pars =  {
                    'e_rev_E': 0.0,
                    'tau_m': 20.0,
                    'cm': 1.0,
                    'v_thresh': -50.0,
                    'v_rest': -65.0,
                    'i_offset': 0.0,
                    'tau_syn_I': 0.5,
                    'tau_syn_E': 0.5, # std 0.3
                    'tau_refrac': 0.1,
                    'v_reset': -65.0,
                    'e_rev_I': -70.0}

            ###---DESIGNING THE 2D BUMP ATTRACTOR NETWORK-------------------------------------------------------
            netSpace = Grid2D(aspect_ratio=1,
                                dx = 1.0, # TODO: pick the distance on the cortex in mm
                                dy = 1.0, # TODO: pick the distance on the cortex in mm
                                x0 = 0.0,
                                y0 = 0.0,
                                fill_order = "sequential")

            #Excitatory cells
            nCells = self.args.N_pop #30X30 surface
            #print("cells_positions", netSpace.generate_positions(nCells))
            #print("cells_size", netSpace.calculate_size(nCells))
            popE = sim.Population(nCells,
                sim.IF_cond_exp,
                cell_pars,
                structure = netSpace,
                label="2D-bumpNN")
            popE.record(['spikes','v','gsyn_exc','gsyn_inh'])
            if self.args.verbose: print("pop", popE)

            #Inhibitory cells
            nCells = self.args.N_pop #30X30 surface
            popI = sim.Population(nCells,
                sim.IF_cond_exp,
                cell_pars,
                structure = netSpace,
                label="2D-bumpNN")
            popI.record(['spikes','v','gsyn_exc','gsyn_inh'])
            if self.args.verbose: print("pop", popI)


            def connectionGenerator(pop1, pop2, connType, weightFun, scalePar, waveVelocity):
                connList = []
                #Distance Based Probability Connection
                if type(connType) == float:
                    print('connType is float')
                    p = []
                    par =  connType
                    for i in range(pop1):
                        for j in range(pop2):
                            Ax = i // int(np.sqrt(pop1))
                            Ay = i % int(np.sqrt(pop1))
                            Bx = j // int(np.sqrt(pop2))
                            By = j % int(np.sqrt(pop2))

                            d = np.sqrt((Ax - Bx)**2 + (Ay - By)**2) #abs(i-j)
                            prob = np.exp(-d**(1/par))
                            p.append(prob)
                            delayFun = round(self.args.min_delay + (scalePar*d)/waveVelocity, 2)
                            delayFun = np.min((delayFun, self.args.max_delay-1))
                            if prob > np.random.random():
                                connList = connList + [(i, j, weightFun, delayFun)]


                else:
                    print('connType is string')
                    if connType == 'oneTOone':
                        for i in range(pop1):
                            d = abs(i-i)
                            delayFun = round(self.args.min_delay + (scalePar*d)/waveVelocity, 2)
                            delayFun = np.min((delayFun, self.args.max_delay-1))
                            connList = connList + [(i, i, round(weightFun,3), delayFun)]

                    if connType == 'oneTOoneDelay30ms':
                        for i in range(pop1):
                            d = abs(i-i)
                            delayFun = 30 #ms
                            connList = connList + [(i, i, round(weightFun,3), delayFun)]

                    print('connType is string')
                    if connType == 'oneTOmany':
                        delta = 2
                        for i in range(pop1):
                            for j in range(pop2):
                                Ax = i // int(np.sqrt(pop1))
                                Ay = i % int(np.sqrt(pop1))
                                Bx = j // int(np.sqrt(pop2))
                                By = j % int(np.sqrt(pop2))
                                d = np.sqrt((Ax - Bx)**2 + (Ay - By)**2) #abs(i-j)
                                delayFun = round(self.args.min_delay + (scalePar*d)/waveVelocity, 2)
                                delayFun = np.min((delayFun, self.args.max_delay-1))
                                if d <= delta:
                                    connList = connList + [(i, j, weightFun, delayFun)]

                    if connType == 'allTOall':
                        for i in range(pop1):
                            for j in range(pop2):
                                Ax = i // int(np.sqrt(pop1))
                                Ay = i % int(np.sqrt(pop1))
                                Bx = j // int(np.sqrt(pop2))
                                By = j % int(np.sqrt(pop2))
                                d = np.sqrt((Ax - Bx)**2 + (Ay - By)**2) #abs(i-j)
                                delayFun = round(self.args.min_delay + (scalePar*d)/waveVelocity, 2)
                                delayFun = np.min((delayFun, self.args.max_delay-1))
                                connList = connList + [(i, j, weightFun, delayFun)]

                    if connType == 'uniform':
                        p = []
                        for i in range(pop1):
                            for j in range(pop2):
                                Ax = i // int(np.sqrt(pop1))
                                Ay = i % int(np.sqrt(pop1))
                                Bx = j // int(np.sqrt(pop2))
                                By = j % int(np.sqrt(pop2))
                                d = np.sqrt((Ax - Bx)**2 + (Ay - By)**2) #abs(i-j)
                                prob = np.random.uniform(0,1)
                                p.append(prob)
                                delayFun = round(self.args.min_delay + (scalePar*d)/waveVelocity, 2)
                                delayFun = np.min((delayFun, self.args.max_delay-1))
                                if prob > 0.50:
                                    connList = connList + [(i, j, weightFun, delayFun)]


                return connList


            ###--- CONNECTIONS IN THE 2D BUMP NETWORK ---
            connEE = connectionGenerator(self.args.N_pop, self.args.N_pop, self.args.E_expPar, self.args.E_weight, self.args.scalePar, self.args.waveVelocity)
            #connEI = connectionGenerator(self.args.N_pop, self.args.N_pop, self.args.E_expPar, self.args.E_weight, self.args.scalePar, self.args.waveVelocity)
            connEIone = connectionGenerator(self.args.N_pop, self.args.N_pop, 'oneTOoneDelay30ms', self.args.Eone_weight, self.args.scalePar, self.args.waveVelocity)
            connII = connectionGenerator(self.args.N_pop, self.args.N_pop, self.args.I_expPar, self.args.I_weight, self.args.scalePar, self.args.waveVelocity)
            #connIE = connectionGenerator(self.args.N_pop, self.args.N_pop, self.args.I_expPar, self.args.I_weight, self.args.scalePar, self.args.waveVelocity)
            connIEone = connectionGenerator(self.args.N_pop, self.args.N_pop, 'oneTOoneDelay30ms', self.args.Ione_weight, self.args.scalePar, self.args.waveVelocity)

            # making the projections
            self.projEE = sim.Projection(popE, popE,
                connector = sim.FromListConnector(connEE),
                receptor_type = "excitatory",
                label = "exc_connections")

            #self.projEI = sim.Projection(popE, popI,
            #    connector = sim.FromListConnector(connEI),
            #    receptor_type = "excitatory",
            #    label = "exc_connections")

            self.projEIone = sim.Projection(popE, popI,
                connector = sim.FromListConnector(connEIone),
                receptor_type = "excitatory",
                label = "exc_connections")

            self.projII = sim.Projection(popI, popI,
                connector = sim.FromListConnector(connII),
                receptor_type = "inhibitory",
                label = "inh_connections")

            #self.projIE = sim.Projection(popI, popE,
            #    connector = sim.FromListConnector(connIE),
            #    receptor_type = "inhibitory",
            #    label = "inh_connections")

            self.projIEone = sim.Projection(popI, popE,
                connector = sim.FromListConnector(connIEone),
                receptor_type = "inhibitory",
                label = "inh_connections")


            # making the sources projections
            sources = sim.SpikeSourceArray(spike_times=cellSourceSpikes)
            spikeSource = sim.Population(self.args.N_pop, sources)
            spikeSource.record(['spikes'])
            scaleDistanceFactor = 0 # this allows to have the total delay equal to delay_min)

            self.projSourceE = sim.Projection(spikeSource, popE,
                        connector = sim.FromListConnector(connectionGenerator(self.args.N_pop,
                                                                              self.args.N_pop,
                                                                              'oneTOmany',
                                                                              0.1,
                                                                              scaleDistanceFactor,
                                                                              self.args.waveVelocity)),
                        receptor_type = "excitatory",
                        label = "source_connections")





            #if self.args.verbose: print(list(connectors), projSource)

            #print(list(connectors))
            ###---SIM FINAL STEP--------------------------------------------------------------------------------
            sim.run(self.time_bins)

            ###---SAVE PKL DATA---------------------------------------------------------------------------------
            self.popEdata = popE.get_data()
            self.popIdata = popI.get_data()
            self.spikeSources = spikeSource.get_data()

            if self.args.do_save:
                #Save popE and popI particular data
                # popE.write_data("outputs/input_2D/prob/E_spikes.pkl",'spikes')
                # popE.write_data("outputs/input_2D/prob/E_voltage.pkl",'v')
                # popE.write_data("outputs/input_2D/prob/E_gsyn_exc.pkl",'gsyn_exc')
                # popE.write_data("outputs/input_2D/prob/E_gsyn_inh.pkl",'gsyn_inh')
                # popI.write_data("outputs/input_2D/prob/I_spikes.pkl",'spikes')
                # popI.write_data("outputs/input_2D/prob/I_voltage.pkl",'v')
                # popI.write_data("outputs/input_2D/prob/I_gsyn_exc.pkl",'gsyn_exc')
                # popI.write_data("outputs/input_2D/prob/I_gsyn_inh.pkl",'gsyn_inh')

                #Save popE and popI global data
                pickle.dump(self.popEdata, open(os.path.join(self.tagpath, 'popEdata.pkl'), 'wb'))
                pickle.dump(self.popIdata, open(os.path.join(self.tagpath, 'popIdata.pkl'), 'wb'))
                pickle.dump(self.spikeSources, open(os.path.join(self.tagpath, 'spikeSource.pkl'),'wb'))

            if self.args.verbose:
                for i in range(3):
                    print(self.popEdata.segments[0].analogsignals[i].name)
                    print(self.popIdata.segments[0].analogsignals[i].name)

            if self.args.do_save:
                ###---SAVE PROJECTIONS (weights and delay)-----------------------------------------------------------
                self.projSourceE.save(['weight', 'delay'],os.path.join(self.tagpath, 'weights_projSourceE.txt'), format='list')
                self.projEE.save(['weight', 'delay'],os.path.join(self.tagpath, 'weights_projEE.txt'), format='list')
                #self.projEI.save(['weight', 'delay'], os.path.join(self.tagpath, 'weights_projEI.txt'), format='list')
                self.projEIone.save(['weight','delay'], os.path.join(self.tagpath, 'weights_projEIone.txt'), format='list')
                self.projII.save(['weight', 'delay'],os.path.join(self.tagpath, 'weights_projII.txt'), format='list')
                #self.projIE.save(['weight', 'delay'], os.path.join(self.tagpath, 'weights_projIE.txt'), format='list')
                self.projIEone.save(['weight', 'delay'], os.path.join(self.tagpath, 'weights_projIEone.txt'), format='list')

            sim.end()


        dataEseg = self.popEdata.segments[0]
        dataIseg = self.popIdata.segments[0]

        for i in range(3):
            # E layer
            if dataEseg.analogsignals[i].name == 'v':
                E_voltage = dataEseg.analogsignals[i]
            if dataEseg.analogsignals[i].name == 'gsyn_exc':
                E_gsynE = dataEseg.analogsignals[i]
            if dataEseg.analogsignals[i].name == 'gsyn_inh':
                E_gsynI = dataEseg.analogsignals[i]
            # I layer
            if dataIseg.analogsignals[i].name == 'v':
                I_voltage = dataIseg.analogsignals[i]
            if dataIseg.analogsignals[i].name == 'gsyn_exc':
                I_gsynE = dataIseg.analogsignals[i]
            if dataIseg.analogsignals[i].name == 'gsyn_inh':
                I_gsynI = dataIseg.analogsignals[i]

        if self.args.verbose:
            print(E_voltage.name)
            print(E_gsynE.name)
            print(E_gsynI.name)
            print(I_voltage.name)
            print(I_gsynE.name)
            print(I_gsynI.name)

            
        self.E_spikes = self.popEdata.segments[0].spiketrains
        self.I_spikes = self.popIdata.segments[0].spiketrains
        self.S_spikes = self.spikeSources.segments[0].spiketrains

        self.I_v = np.transpose(I_voltage)
        self.E_v = np.transpose(E_voltage)
        self.E_gsynE = np.transpose(E_gsynE)
        self.E_gsynI = np.transpose(E_gsynI)
        self.I_gsynE = np.transpose(I_gsynE)
        self.I_gsynI = np.transpose(I_gsynI)

        self.dim_shape = int(np.sqrt(self.args.N_pop))
        self.time_bins = int(self.args.simtime/self.args.dt) #extended time
        
        return [self.S_spikes,
                self.E_spikes, 
                self.E_v, 
                self.E_gsynE, 
                self.E_gsynI, 
                self.I_spikes, 
                self.I_v, 
                self.I_gsynE, 
                self.I_gsynI,                
                self.dim_shape, 
                self.time_bins]


# >>>>>>>>> ============= STOP CODE HERE =============== <<<<<<<<<<<<<<<<<<<


    def plot_spikes(self, showArg=True):

        fig = plt.figure(figsize=(14, 12))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        #ax1 = fig.add_subplot(3,1,1)
        #im1 = plt.eventplot(inputs, colors='green')
        #ax1.set_title("Input Spikes", y=1.10, fontsize=16)
        #ax1.set_xlabel("[ms]", fontsize=14);
        #ax1.set_ylabel("neurons", fontsize=14)
        #ax1.set_xticks(range(0,1000,250))
        #TO DO: MODIFY THE SCRIPT TO PRINT THE COMPLETE (NOT PARTIAL) SEQUENCE OF INPUT SPIKES

        ax2 = fig.add_subplot(3,1,1)
        im2 = plt.eventplot(self.spikeSources.segments[0].spiketrains, colors='black')
        ax2.set_title("Spikes in Sources Layer ", y=1.10, fontsize=16)
        ax2.set_xlabel("[ms]", fontsize=14);
        ax2.set_ylabel("neurons", fontsize=14)

        ax2 = fig.add_subplot(3,1,2)
        im2 = plt.eventplot(self.popEdata.segments[0].spiketrains, colors='red')
        ax2.set_title("Spikes in E Layer ", y=1.10, fontsize=16)
        ax2.set_xlabel("[ms]", fontsize=14);
        ax2.set_ylabel("neurons", fontsize=14)

        ax3 = fig.add_subplot(3,1,3)
        im3 = plt.eventplot(self.popIdata.segments[0].spiketrains, colors='blue')
        ax3.set_title("Spikes in I Layer ", y=1.10, fontsize=16)
        ax3.set_xlabel("[ms]", fontsize=14);
        ax3.set_ylabel("neurons", fontsize=14)

        fig.suptitle("Spike trains in E and I layers", fontsize=18)
        fig.subplots_adjust(hspace=0.5,wspace=0.4)

        fig.savefig(os.path.join(self.figpath, 'spikes.pdf'))



    def plot_firingrates(self, showArg=True):
        S_spikes = self.spikeSources.segments[0].spiketrains
        E_spikes = self.popEdata.segments[0].spiketrains
        I_spikes = self.popIdata.segments[0].spiketrains

    # Computing the firing rates
        N_pop = self.args.N_pop
        time_bins = self.time_bins
        N_tau = self.args.N_tau
        dt = 1/ time_bins
        def firing_rate(b, N_tau=N_tau) :
            fr = np.zeros_like(b).astype(float)
            for t in range(b.shape[0]) :
            #print(1 - 1 / tau, t, fr.mean())
                fr[t, :] = (1 - 1 / N_tau) * fr[t-1, :] + 1 / N_tau * b[t, :]
            return fr.T # transposition to have (N,T)

        # FIRING RATES OF SOURCES SPIKES
        S_binMat = np.zeros((time_bins, N_pop), dtype='int')
        S_spk_list = []
        for i in range(N_pop): # loops over sources
            S_spike_train = S_spikes[i].as_array() # makes as array each source
            S_spike_times = [int(t) for t in S_spike_train]
            S_spk_list.append(S_spikes[i].as_array())
            for spike_time in S_spike_times: # loops over spikes for each source
                    #print(spike_time, i)
                    S_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                    # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        S_fr = firing_rate(S_binMat)*(1/dt)

        # pcolor plots
        rows = 1
        cols = 3
        fig, axes = plt.subplots(rows,cols,figsize=(14,5))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        fig.tight_layout(pad=3.0)
        axes_list = fig.axes

        S_spk_varFR = []
        for i in range(len(S_fr)):
            S_spk_varFR.append(np.var(S_fr[i]))
        S_spk_varFR = np.asarray(S_spk_varFR).reshape((int(np.sqrt(len(self.E_v))),int(np.sqrt(len(self.E_v)))))

        data_list = [S_binMat.T, S_fr, S_spk_varFR]
        titles_list = ['binarized input spike sources [0/1]',
                      'input spike sources fr [Hz]',
                      'variance of input spike sources fr']

        for i in range(rows*cols):
            im = axes_list[i].pcolor(data_list[i]); fig.colorbar(im,ax=axes_list[i])
            #axes_list[i].set_ylim(0, N_pop)
            axes_list[i].set_title(titles_list[i], fontsize=16)

        fig.savefig(os.path.join(self.figpath, 'firingrates_sources.pdf'))


        # FIRING RATES OF E LAYER SPIKES
        E_binMat = np.zeros((time_bins, N_pop), dtype='int')
        E_spk_list = []
        for i in range(N_pop): # loops over sources
            E_spike_train = E_spikes[i].as_array() # makes as array each source
            E_spk_list.append(E_spikes[i].as_array())
            E_spike_times = [int(t) for t in E_spike_train]
            for spike_time in E_spike_times: # loops over spikes for each source
                #print(spike_time, i)
                E_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        E_fr = firing_rate(E_binMat)*(1/dt)

        # pcolor plots
        rows = 1
        cols = 3
        fig, axes = plt.subplots(rows,cols,figsize=(14,5))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()
        fig.tight_layout(pad=3.0)
        axes_list = fig.axes

        E_spk_varFR = []
        for i in range(len(E_fr)):
            E_spk_varFR.append(np.var(E_fr[i]))
        E_spk_varFR = np.asarray(E_spk_varFR).reshape(int(np.sqrt(len(self.E_v))),int(np.sqrt(len(self.E_v))))

        data_list = [E_binMat.T, E_fr, E_spk_varFR]
        titles_list = ['binarized E spikes [0/1]',
                      'E spikes fr [Hz]',
                      'variance of E spikes fr']

        for i in range(rows*cols):
            im = axes_list[i].pcolor(data_list[i]); fig.colorbar(im,ax=axes_list[i])
            #axes_list[i].set_ylim(0, N_pop)
            axes_list[i].set_title(titles_list[i], fontsize=16)

        fig.savefig(os.path.join(self.figpath, 'firingrates_E_layer.pdf'))


        # FIRING RATES OF I LAYER SPIKES
        I_binMat = np.zeros((time_bins, N_pop), dtype='int')
        I_spk_list = []
        for i in range(N_pop): # loops over sources
            I_spike_train = I_spikes[i].as_array() # makes as array each source
            I_spk_list.append(I_spikes[i].as_array())
            I_spike_times = [int(t) for t in I_spike_train]
            for spike_time in I_spike_times: # loops over spikes for each source
                #print(spike_time, i)
                I_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        I_fr = firing_rate(I_binMat)*(1/dt)

        # pcolor plots
        rows = 1
        cols = 3
        fig, axes = plt.subplots(rows,cols,figsize=(14,5))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        fig.tight_layout(pad=3.0)
        axes_list = fig.axes

        I_spk_varFR = []
        for i in range(len(I_fr)):
            I_spk_varFR.append(np.var(I_fr[i]))
        I_spk_varFR = np.asarray(I_spk_varFR).reshape(int(np.sqrt(len(self.I_v))),int(np.sqrt(len(self.I_v))))

        data_list = [I_binMat.T, I_fr, I_spk_varFR]
        titles_list = ['binarized I spikes [0/1]',
                      'I spikes fr [Hz]',
                      'variance of I spikes fr']

        for i in range(rows*cols):
            im = axes_list[i].pcolor(data_list[i]); fig.colorbar(im,ax=axes_list[i])
            #axes_list[i].set_ylim(0, N_pop)
            axes_list[i].set_title(titles_list[i], fontsize=16)

        fig.savefig(os.path.join(self.figpath, 'firingrates_I_layer.pdf'))

        self.S_fr = S_fr
        self.E_fr = E_fr
        self.I_fr = I_fr
        return self.S_fr, self.E_fr, self.I_fr

    def plot_voltage(self, showArg=True):
        ###---PLOTS OF VOLTAGE TIME COURSE AND VOLTAGE VARIANCE OF EXCITATORY AND INHIBITORY LAYER

        fig = plt.figure(figsize=(12,12))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        ax1 = fig.add_subplot(4,2,1)
        im1 = ax1.matshow(self.E_v); fig.colorbar(im1)
        ax1.set_title("Voltage in E Layer [mV]", y=1.20, fontsize=12)
        ax1.set_xlabel("[ms]", fontsize=16);
        ax1.set_ylabel("neurons", fontsize=16)
        ax3 = fig.add_subplot(4,2,3)

        E_varV = []
        for i in range(len(self.E_v)):
            E_varV.append([np.var(self.E_v[i])])
        E_varV=np.asarray(E_varV)
        E_varV.resize(int(np.sqrt(len(self.E_v))),int(np.sqrt(len(self.E_v))))
        im3=ax3.matshow(E_varV); plt.colorbar(im3)
        ax3.set_title("Voltage Variance in E Layer", y=1.10, fontsize=12)
        ax3.set_xlabel("neurons", fontsize=16); ax3.set_xticks([])
        ax3.set_ylabel("neurons", fontsize=16)

        ax5 = fig.add_subplot(4,2,5)
        im5 = ax5.matshow(np.cov(self.E_v)); fig.colorbar(im5)
        ax5.set_title("Voltage Covariance Matrix in E Layer", y=1.10, fontsize=12)
        ax5.set_xlabel("neurons", fontsize=16); ax5.set_xticks([])
        ax5.set_ylabel("neurons", fontsize=16)

        ax7 = fig.add_subplot(4,2,7)
        im7 = ax7.matshow(np.corrcoef(self.E_v)); fig.colorbar(im7)
        ax7.set_title("Voltage Correlation Matrix in E Layer ", y=1.10, fontsize=12)
        ax7.set_xlabel("neurons", fontsize=16); ax7.set_xticks([])
        ax7.set_ylabel("neurons", fontsize=16)

        ax2 = fig.add_subplot(4,2,2)
        im2 = ax2.matshow(self.I_v); fig.colorbar(im2)
        ax2.set_title("Voltage in I Layer [mV]", y=1.20, fontsize=12)
        ax2.set_xlabel("[ms]", fontsize=16);
        ax2.set_ylabel("neurons", fontsize=16)

        I_varV = []
        for i in range(len(self.I_v)):
            I_varV.append([np.var(self.I_v[i])])
        I_varV=np.asarray(I_varV)
        I_varV.resize(int(np.sqrt(len(self.I_v))),int(np.sqrt(len(self.I_v))))
        ax4 = fig.add_subplot(4,2,4)
        im4 = ax4.matshow(I_varV); fig.colorbar(im4)
        ax4.set_title("Voltage Variance in I Layer", y=1.10, fontsize=12)
        ax4.set_xlabel("neurons", fontsize=16); ax4.set_xticks([])
        ax4.set_ylabel("neurons", fontsize=16)

        ax6 = fig.add_subplot(4,2,6)
        im6 = ax6.matshow(np.cov(self.I_v)); fig.colorbar(im6)
        ax6.set_title("Voltage Covariance Matrix in I Layer ", y=1.10, fontsize=12)
        ax6.set_xlabel("neurons", fontsize=16); ax6.set_xticks([])
        ax6.set_ylabel("neurons", fontsize=16)

        ax8 = fig.add_subplot(4,2,8)
        im8 = ax8.matshow(np.corrcoef(self.I_v)); fig.colorbar(im8)
        ax8.set_title("Voltage Correlation Matrix in I Layer ", y=1.10, fontsize=12)
        ax8.set_xlabel("neurons", fontsize=16); ax8.set_xticks([])
        ax8.set_ylabel("neurons", fontsize=16)

        fig.suptitle("Voltage Description in E and I layers", fontsize=18)
        fig.subplots_adjust(hspace=0.5,wspace=0.4)

        fig.savefig(os.path.join(self.figpath, 'voltage.pdf'))

    def plot_crossedclustering(self, PCA_representation, showArg=True):
    ###---CROSSED-CLUSTERINIG ANALYSIS
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score as db


        #Excitatory layer
        E_NOI_pred = []
        E_TOI_pred = []
        E_TOI_pred_DB = []
        E_NOI_pred_DB = []
        E_clusters = np.arange(2,20)

        for cluster in E_clusters:
            E_NOI_pred = E_NOI_pred + [KMeans(n_clusters=cluster).fit_predict(self.E_v)]
            E_TOI_pred = E_TOI_pred + [KMeans(n_clusters=cluster).fit_predict(np.transpose(self.E_v))]
            E_NOI_pred_DB = E_NOI_pred_DB + [db(self.E_v, E_NOI_pred[cluster-2])]
            E_TOI_pred_DB = E_TOI_pred_DB + [db(np.transpose(self.E_v), E_TOI_pred[cluster-2])]

        print("Optimal number of Neuron Of Interest (NOI) Clusters in E layer:", E_NOI_pred_DB.index(min(E_NOI_pred_DB))+2)
        print("Optimal number of Time Of Interest (TOI) Clusters in E layer:", E_TOI_pred_DB.index(min(E_TOI_pred_DB))+2)

        #Inhibitory layer
        I_NOI_pred = []
        I_TOI_pred = []
        I_TOI_pred_DB = []
        I_NOI_pred_DB = []
        I_clusters = np.arange(2,20)

        for cluster in I_clusters:
            I_NOI_pred = I_NOI_pred + [KMeans(n_clusters=cluster).fit_predict(self.I_v)]
            I_TOI_pred = I_TOI_pred + [KMeans(n_clusters=cluster).fit_predict(np.transpose(self.I_v))]
            I_NOI_pred_DB = I_NOI_pred_DB + [db(self.I_v, I_NOI_pred[cluster-2])]
            I_TOI_pred_DB = I_TOI_pred_DB + [db(np.transpose(self.I_v), I_TOI_pred[cluster-2])]

        print("Optimal number of Neuron Of Interest (NOI) Clusters in I layer:", I_NOI_pred_DB.index(min(I_NOI_pred_DB))+2)
        print("Optimal number of Time Of Interest (TOI) Clusters in I layer:", I_TOI_pred_DB.index(min(I_TOI_pred_DB))+2)

        fig = plt.figure(figsize=(12,10))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        ax2 = fig.add_subplot(3,4,2)
        xrange = np.arange(2,len(E_NOI_pred_DB)+2)
        plt.plot(xrange,E_NOI_pred_DB, label="NOI clustering");
        plt.plot(E_NOI_pred_DB.index(min(E_NOI_pred_DB))+2,min(E_NOI_pred_DB),'ro',label='optimal NOI cluster')
        plt.plot(xrange,E_TOI_pred_DB, label="TOI clustering");
        plt.plot(E_TOI_pred_DB.index(min(E_TOI_pred_DB))+2,min(E_TOI_pred_DB),'go', label='optimal TOI cluster')
        ax2.set_title("Clustering validation \n E layer", y=1.10, fontsize=12)
        ax2.set_xlabel("Clusters", fontsize=12); ax2.set_xticks(np.arange(0,len(E_NOI_pred)+2,1));
        ax2.set_ylabel("DBI", fontsize=12)
        ax2.set(xlim=(0,20),ylim=(0,10))
        ax2.legend(); ax2.grid(True)

        ax3 = fig.add_subplot(3,4,3)
        xrange = np.arange(2,len(I_NOI_pred_DB)+2)
        plt.plot(xrange,I_NOI_pred_DB, label="NOI clustering");
        plt.plot(I_NOI_pred_DB.index(min(I_NOI_pred_DB))+2,min(I_NOI_pred_DB),'ro',label='optimal NOI cluster')
        plt.plot(xrange,I_TOI_pred_DB, label="TOI clustering");
        plt.plot(I_TOI_pred_DB.index(min(I_TOI_pred_DB))+2,min(I_TOI_pred_DB),'go', label='optimal TOI cluster')
        ax3.set_title("Clustering validation \n I layer", y=1.10, fontsize=12)
        ax3.set_xlabel("Clusters", fontsize=12); ax3.set_xticks(np.arange(0,len(I_NOI_pred)+2,1));
        ax3.set_ylabel("DBI", fontsize=12)
        ax3.set(xlim=(0,20),ylim=(0,10))
        ax3.legend(); ax3.grid(True)

        ax5 = fig.add_subplot(3,4,5)
        E_NOI = np.zeros((len(self.E_v),len(self.E_v[0])))
        E_NOI_labels=E_NOI_pred[E_NOI_pred_DB.index(min(E_NOI_pred_DB))]
        for i in np.arange(len(E_NOI[:])):
            for j in np.arange(len(E_NOI[0])):
                E_NOI[i][j] = E_NOI[i][j] + [E_NOI_labels[i]]
        im5=ax5.matshow(E_NOI); fig.colorbar(im5, fraction=0.035, pad=0.04)
        ax5.set_title("NOI clustering \n Voltage E Layer [clusters]", y=1.20, fontsize=12)
        ax5.set_xlabel("[ms]", fontsize=12);
        ax5.set_ylabel("neurons", fontsize=12)

        ax6 = fig.add_subplot(3,4,6)
        im6 = ax6.matshow(self.E_v); fig.colorbar(im6,fraction=0.035, pad=0.04)
        ax6.set_title("Voltage in E Layer [mV]", y=1.20, fontsize=12)
        ax6.set_xlabel("[ms]", fontsize=12);
        ax6.set_ylabel("neurons", fontsize=12)

        ax7 = fig.add_subplot(3,4,7)
        im7 = ax7.matshow(self.I_v); fig.colorbar(im7,fraction=0.035, pad=0.04)
        ax7.set_title("Voltage in I Layer [mV]", y=1.20, fontsize=12)
        ax7.set_xlabel("[ms]", fontsize=12);
        ax7.set_ylabel("neurons", fontsize=12)

        ax8 = fig.add_subplot(3,4,8)
        I_NOI = np.zeros((len(self.I_v),len(self.I_v[0])))
        I_NOI_labels=I_NOI_pred[I_NOI_pred_DB.index(min(I_NOI_pred_DB))]
        for i in np.arange(len(I_NOI[:])):
            for j in np.arange(len(I_NOI[0])):
                I_NOI[i][j] = I_NOI[i][j] + [I_NOI_labels[i]]
        im8=ax8.matshow(I_NOI); fig.colorbar(im8, fraction=0.035, pad=0.04)
        ax8.set_title("NOI clustering \n Voltage I Layer [clusters]", y=1.20, fontsize=12)
        ax8.set_xlabel("[ms]", fontsize=12);
        ax8.set_ylabel("neurons", fontsize=12)

        #CHECK BOTTOM CODE

        ax11 = fig.add_subplot(3,4,11)
        I_TOI = np.zeros((len(self.I_v),len(self.I_v[0])))
        I_TOI_labels=I_TOI_pred[I_TOI_pred_DB.index(min(I_TOI_pred_DB))]
        for i in np.arange(len(I_TOI[:])):
            for j in np.arange(len(I_TOI[0])):
                I_TOI[i][j] = I_TOI[i][j] + [I_TOI_labels[i]]
        im11=ax11.matshow(np.transpose(I_TOI));
        fig.colorbar(im11, fraction=0.035, pad=0.04)
        ax11.set_title("TOI clustering \n Voltage I Layer \n [clusters]", y=1.20, fontsize=12)
        ax11.set_xlabel("[ms]", fontsize=12);
        ax11.set_ylabel("neurons", fontsize=12)


        ax10 = fig.add_subplot(3,4,10)
        E_TOI = np.zeros((len(self.E_v),len(self.E_v[0])))
        E_TOI_labels=E_TOI_pred[E_TOI_pred_DB.index(min(E_TOI_pred_DB))]
        for i in np.arange(len(E_TOI[:])):
            for j in np.arange(len(E_TOI[0])):
                E_TOI[i][j] = E_TOI[i][j] + [E_TOI_labels[i]]
        im10=ax10.matshow(np.transpose(E_TOI));
        fig.colorbar(im10, fraction=0.035, pad=0.04)
        ax10.set_title("TOI clustering \n Voltage E Layer \n [clusters]", y=1.20, fontsize=12)
        ax10.set_xlabel("[ms]", fontsize=12);
        ax10.set_ylabel("neurons", fontsize=12)

        fig.subplots_adjust(hspace=0.5,wspace=0.7)

        fig.savefig(os.path.join(self.figpath, 'crossclustering_1.pdf'))


        if PCA_representation == "True":
            ### PCA analysis of E layer
            from sklearn.decomposition import PCA
            dim_comps = 100

            TOI_pca = PCA(n_components=dim_comps, svd_solver='full').fit(self.E_v)
            TOI_eig = TOI_pca.components_

            NOI_pca = PCA(n_components=dim_comps, svd_solver='full').fit(np.transpose(self.E_v))
            NOI_eig = NOI_pca.components_

            optNOI = E_NOI_pred_DB.index(min(E_NOI_pred_DB))
            optTOI = E_TOI_pred_DB.index(min(E_TOI_pred_DB))

            fig = plt.figure(figsize=(18,14))
            if showArg == 'False':
                plt.close(fig)
            if showArg == 'True':
                plt.show()

            ax1 = fig.add_subplot(4,2,1)
            im1 = ax1.semilogy(TOI_pca.explained_variance_)
            ax1.hlines(1,0,dim_comps, label="lambda = 1")
            ax1.set_title("Time Of Interest (TOI) ", y=1.10, fontsize=16)
            ax1.set_xlabel("Eigenvectors of E voltage", fontsize=16)
            ax1.set_ylabel("Explaned variance", fontsize=16)

            ax2 = fig.add_subplot(4,2,2)
            im2 = ax2.semilogy(NOI_pca.explained_variance_)
            ax2.hlines(1,0,dim_comps, label="lambda = 1")
            ax2.set_title("Neuron Of Interest (NOI) ", y=1.10, fontsize=16)
            ax2.set_xlabel("Eigenvectors of E voltage", fontsize=16)
            ax2.set_ylabel("Explaned variance", fontsize=16)

            ax3 = fig.add_subplot(4,2,3)
            im3 = ax3.scatter(TOI_eig[0],TOI_eig[1], c=E_TOI_pred[optTOI])
            cbar = fig.colorbar(im3); cbar.ax.set_ylabel('Clusters')
            ax3.set_title("TOI Clusters on data-reduced of E voltage", y=1.10, fontsize=16)
            ax3.set_xlabel("1st Eigenvector", fontsize=16)
            ax3.set_ylabel("2nd Eigenvector", fontsize=16)

            ax4 = fig.add_subplot(4,2,4)
            im4 = ax4.scatter(NOI_eig[0],NOI_eig[1], c=E_NOI_pred[optNOI])
            cbar = fig.colorbar(im4); cbar.ax.set_ylabel('Clusters')
            ax4.set_title("NOI Clusters on data-reduced of E voltage", y=1.10, fontsize=16)
            ax4.set_xlabel("1st Eigenvector", fontsize=16)
            ax4.set_ylabel("2nd Eigenvector", fontsize=16)

            ax5 = fig.add_subplot(4,2,5, projection='3d')
            im5 = ax5.scatter(TOI_eig[0],TOI_eig[1],TOI_eig[2],c=E_TOI_pred[optTOI]);
            cbar = fig.colorbar(im5); cbar.ax.set_ylabel('Clusters')
            ax5.set_title("1st, 2nd and 3rd eigenvecotors", y=1.10, fontsize=16)
            #ax5.set_xlabel("1st Eigenvector", fontsize=16)
            #ax5.set_ylabel("2nd Eigenvector", fontsize=16)
            #ax5.set_zlabel("3rd Eigenvector", fontsize=16)

            ax6 = fig.add_subplot(4,2,6, projection='3d')
            im6 = ax6.scatter(NOI_eig[0],NOI_eig[1],NOI_eig[2],c=E_NOI_pred[optNOI])
            cbar = fig.colorbar(im6); cbar.ax.set_ylabel('Clusters')
            ax6.set_title("1st, 2nd and 3rd eigenvecotors", y=1.10, fontsize=16)
            #ax6.axis('off')
            #ax6.set_xlabel("1st Eigenvector", fontsize=16)
            #ax6.set_ylabel("2nd Eigenvector", fontsize=16)
            #ax6.set_zlabel("3rd Eigenvector", fontsize=16)

            ax7 = fig.add_subplot(4,2,7)
            for i in range(3):
                im7 = ax7.plot(TOI_pca.components_[i],label="TOI eig_{}".format(i))
                ax7.set_title("Scores of first 3 eigenvectors (TOI)", y=1.10, fontsize=16)
                ax7.legend()

            ax8 = fig.add_subplot(4,2,8)
            for i in range(3):
                im8 = ax8.plot(NOI_pca.components_[i],label="NOI eig_{}".format(i))
                ax8.set_title("Scores of first 3 eigenvectors (NOI)", y=1.10, fontsize=16)
                ax8.legend()

            fig.suptitle("PCA data reduction and representation of NOI and TOI optimal clustering in E layer", fontsize=18)
            fig.subplots_adjust(hspace=0.6,wspace=0.2)

            print(len(TOI_eig[0]))
            print(len(NOI_eig[0]))
            fig.savefig(os.path.join(self.figpath, 'crossclustering_2.pdf'))


            ###-------------------------------------------------------------------------------------------------------------###
            # arxiv.org/pdf/1306.3825.pdf Burda et al 2013

            #rho(lambda)=1/(2*pi*r*lambda) * sqrt((lambda+ - lambda)*(lambda- - lambda))
            #r = N/T ; N=neurons; T=times
            #lambda = [lambda-, lambda+]
            #lambda+/- = (1+/-sqrt(r))**2

            def MPD(lm, lp, r):
                d = []
                e = np.arange(lm,lp,0.1)
                for i in np.arange(len(e)):
                    l = 1/(np.pi*e[i]*r)
                    s = np.sqrt((lp-e[i])*(e[i]-lm))
                    d.append(l*s)
                return d, e

            ###-------------------------------------------------------------------------------------------------------------###

            len(TOI_pca.explained_variance_)
            len(TOI_pca.components_[0])
            len(NOI_pca.components_[0])
            #NOI
            print('Marchenko-Pastur Disribution parameters for NOI')
            NOI_V, NOI_S = self.E_v.shape
            print('V',NOI_V); print('S',NOI_S)
            NOI_r = NOI_V/NOI_S; print('r:', NOI_r)
            NOI_lp = (1 + np.sqrt(NOI_r))**2; print('lp:', NOI_lp)
            NOI_lm = (1 - np.sqrt(NOI_r))**2; print('lm:', NOI_lm);
            NOI_EV = NOI_pca.explained_variance_
            print("Non-ranodm NOI eigenvectos: ", len(NOI_EV[NOI_EV>NOI_lp]))
            print('')

            #TOI
            print('Marchenko-Pastur Disribution parameters for TOI')
            TOI_V, TOI_S = np.transpose(self.E_v).shape
            print('V', TOI_V); print('S', TOI_S)
            TOI_r = TOI_V/TOI_S; print('r:', TOI_r)
            TOI_lp = (1 + np.sqrt(TOI_r))**2; print('lp:', TOI_lp)
            TOI_lm = (1 - np.sqrt(TOI_r))**2; print('lm:', TOI_lm)
            TOI_EV = TOI_pca.explained_variance_
            print("Non-random TOI eigenvectors: ", len(TOI_EV[TOI_EV>TOI_lp]))


            ###-------------------------------------------------------------------------------------------------------###
            fig = plt.figure(figsize=(18,14))
            if showArg == 'False':
                plt.close(fig)
            if showArg == 'True':
                plt.show()

            fig.subplots_adjust(hspace=0.75,wspace=0.5)
            ax1 = fig.add_subplot(4,4,1)
            NOI_d, NOI_e = MPD(NOI_lm, NOI_lp, NOI_r)
            im1 = ax1.semilogy(NOI_d, NOI_e, color='r')
            ax1.set_title("Marchenko-Pastur Distribution \n Time Of Interest (NOI)", y=1.10, fontsize=16)
            ax1.set_xlabel("Eigenvalues", fontsize=14)
            ax1.set_ylabel("", fontsize=14)
            ax1.set_ylim(0,self.time_bins)
            ax1.set_xlim(0,3)
            ax1.grid()

            ax2 = fig.add_subplot(4,4,2)
            im2 = ax2.semilogy(NOI_pca.explained_variance_)
            ax2.hlines(NOI_lm,0,dim_comps, label="lambda = lm", color='r')
            ax2.hlines(NOI_lp,0,dim_comps, label="lambda = lp", color='r')
            ax2.set_title("Time Of Interest (NOI) ", y=1.10, fontsize=16)
            ax2.set_xlabel("Eigenvectors of E voltage", fontsize=14)
            ax2.set_ylabel("Eigenvalues", fontsize=14)
            ax2.grid()

            ax3 = fig.add_subplot(4,4,3)
            im3 = ax3.semilogy(TOI_pca.explained_variance_)
            ax3.hlines(TOI_lm,0,dim_comps, label="lambda = lm", color="r")
            ax3.hlines(TOI_lp,0,dim_comps, label="lambda = lp",color="r")
            ax3.set_title("Neuron Of Interest (TOI) ", y=1.10, fontsize=16)
            ax3.set_xlabel("Eigenvectors of E voltage", fontsize=14)
            ax3.set_ylabel("Eigenvalues", fontsize=14)
            ax3.set_ylim(0,self.time_bins)
            ax3.grid()

            ax4 = fig.add_subplot(4,4,4)
            TOI_d, TOI_e = MPD(TOI_lm, TOI_lp, TOI_r)
            im4 = ax4.semilogy(TOI_d, TOI_e, color='r')
            ax4.set_title("Marchenko-Pastur Distribution \n Time Of Interest (TOI)", y=1.10, fontsize=16)
            ax4.set_xlabel("Eigenvalues", fontsize=14)
            ax4.set_ylabel("", fontsize=14)
            ax4.set_ylim(0,self.time_bins)
            ax4.set_xlim(0,3)
            ax4.grid()
            fig.savefig(os.path.join(self.figpath, 'crossclustering_3.pdf'))



            #sns.jointplot(TOI_eig[0],TOI_eig[1])

            #merge the NOI eig 900 x component filter by whishart values
            #plt.matshow(TOI_eig[0])

            #merge the TOI eig 1001 x component filter by whishart values
            #plt.matshow(TOI_eig[0])

            #plot NOI 30X30

            ###-------------------------------------------------------------------------------------------------------------###
            # TOI EIGENVECTORS
            fig=plt.figure()
            if showArg == 'False':
                plt.close(fig)
            if showArg == 'True':
                plt.show()

            im1=plt.matshow(TOI_eig)
            plt.xlabel("[ms]", fontsize=14)
            plt.ylabel("Eigenvectors", fontsize=14)
            plt.title("TOI eigenvectors", fontsize=16)

            ###-------------------------------------------------------------------------------------------------------------###
            # NOI Eigenvectors
            fig, axs = plt.subplots(25,2, figsize=(10,dim_comps))
            #fig.suptitle("NOI Eigenvectors")
            fig.subplots_adjust(hspace = 1, wspace=0.5)
            axs = axs.ravel()

            for i in range(50): #range(len(NOI_EV[NOI_EV>TOI_lp])):
                axs[i].matshow(NOI_eig[i].reshape(self.dim_shape,self.dim_shape))
                axs[i].set_title("NOI Eigenvector " + str(i))
                axs[i].axis('off')

            ### PCA analysis of I layer-------------------------------------------------------------------------------------###

            TOI_pca = PCA(n_components=dim_comps, svd_solver='full').fit(self.I_v)
            TOI_eig = TOI_pca.components_

            NOI_pca = PCA(n_components=dim_comps, svd_solver='full').fit(np.transpose(self.I_v))
            NOI_eig = NOI_pca.components_

            optNOI = I_NOI_pred_DB.index(min(I_NOI_pred_DB))
            optTOI = I_TOI_pred_DB.index(min(I_TOI_pred_DB))

            fig = plt.figure(figsize=(18,14))
            ax1 = fig.add_subplot(4,2,1)
            im1 = ax1.semilogy(TOI_pca.explained_variance_)
            ax1.hlines(1,0,self.args.N_pop, label="lambda = 1")
            ax1.set_title("Time Of Interest (TOI) ", y=1.10, fontsize=16)
            ax1.set_xlabel("Eigenvectors of E voltage", fontsize=16)
            ax1.set_ylabel("Explaned variance", fontsize=16)

            ax2 = fig.add_subplot(4,2,2)
            im2 = ax2.semilogy(NOI_pca.explained_variance_)
            ax2.set_title("Neuron Of Interest (NOI) ", y=1.10, fontsize=16)
            ax2.set_xlabel("Eigenvectors of E voltage", fontsize=16)
            ax2.set_ylabel("Explaned variance", fontsize=16)

            ax3 = fig.add_subplot(4,2,3)
            im3 = ax3.scatter(TOI_eig[0],TOI_eig[1], c=E_TOI_pred[optTOI])
            cbar = fig.colorbar(im3); cbar.ax.set_ylabel('Clusters')
            ax3.set_title("TOI Clusters on data-reduced of I voltage", y=1.10, fontsize=16)
            ax3.set_xlabel("1st Eigenvector", fontsize=16)
            ax3.set_ylabel("2nd Eigenvector", fontsize=16)

            ax4 = fig.add_subplot(4,2,4)
            im4 = ax4.scatter(NOI_eig[0],NOI_eig[1], c=E_NOI_pred[optNOI])
            cbar = fig.colorbar(im4); cbar.ax.set_ylabel('Clusters')
            ax4.set_title("NOI Clusters on data-reduced of I voltage", y=1.10, fontsize=16)
            ax4.set_xlabel("1st Eigenvector", fontsize=16)
            ax4.set_ylabel("2nd Eigenvector", fontsize=16)

            ax5 = fig.add_subplot(4,2,5, projection='3d')
            im5 = ax5.scatter(TOI_eig[0],TOI_eig[1],TOI_eig[2],c=E_TOI_pred[optTOI]);
            cbar = fig.colorbar(im5); cbar.ax.set_ylabel('Clusters')
            ax5.set_title("1st, 2nd and 3rd eigenvecotors", y=1.10, fontsize=16)
            #ax5.set_xlabel("1st Eigenvector", fontsize=16)
            #ax5.set_ylabel("2nd Eigenvector", fontsize=16)
            #ax5.set_zlabel("3rd Eigenvector", fontsize=16)

            ax6 = fig.add_subplot(4,2,6, projection='3d')
            im6 = ax6.scatter(NOI_eig[0],NOI_eig[1],NOI_eig[2],c=E_NOI_pred[optNOI])
            cbar = fig.colorbar(im6); cbar.ax.set_ylabel('Clusters')
            ax6.set_title("1st, 2nd and 3rd eigenvecotors", y=1.10, fontsize=16)
            #ax6.axis('off')
            #ax6.set_xlabel("1st Eigenvector", fontsize=16)
            #ax6.set_ylabel("2nd Eigenvector", fontsize=16)
            #ax6.set_zlabel("3rd Eigenvector", fontsize=16)

            ax7 = fig.add_subplot(4,2,7)
            for i in range(3):
                im7 = ax7.plot(TOI_pca.components_[i],label="TOI eig_{}".format(i))
                ax7.set_title("Scores of first 3 eigenvectors (TOI)", y=1.10, fontsize=16)
                ax7.legend()

            ax8 = fig.add_subplot(4,2,8)
            for i in range(3):
                im8 = ax8.plot(NOI_pca.components_[i],label="NOI eig_{}".format(i))
                ax8.set_title("Scores of first 3 eigenvectors (NOI)", y=1.10, fontsize=16)
                ax8.legend()

            fig.suptitle("PCA data reduction and representation of NOI and TOI optimal clustering in I layer", fontsize=18)
            fig.subplots_adjust(hspace=0.6,wspace=0.2)
            fig.savefig(os.path.join(self.figpath, 'crossclustering_3.pdf'))


            print(len(TOI_eig[0]))
            print(len(NOI_eig[0]))

            len(TOI_pca.explained_variance_)
            len(TOI_pca.components_[0])
            len(NOI_pca.components_[0])
            #NOI
            print('Marchenko-Pastur Disribution parameters for NOI')
            NOI_V, NOI_S = self.I_v.shape
            print('V',NOI_V); print('S',NOI_S)
            NOI_r = NOI_V/NOI_S; print('r:', NOI_r)
            NOI_lp = (1 + np.sqrt(NOI_r))**2; print('lp:', NOI_lp)
            NOI_lm = (1 - np.sqrt(NOI_r))**2; print('lm:', NOI_lm);
            NOI_EV = NOI_pca.explained_variance_
            print("Non-ranodm NOI eigenvectos: ", len(NOI_EV[NOI_EV>NOI_lp]))
            print('')

            #TOI
            print('Marchenko-Pastur Disribution parameters for TOI')
            TOI_V, TOI_S = np.transpose(self.I_v).shape
            print('V', TOI_V); print('S', TOI_S)
            TOI_r = TOI_V/TOI_S; print('r:', TOI_r)
            TOI_lp = (1 + np.sqrt(TOI_r))**2; print('lp:', TOI_lp)
            TOI_lm = (1 - np.sqrt(TOI_r))**2; print('lm:', TOI_lm)
            TOI_EV = TOI_pca.explained_variance_
            print("Non-random TOI eigenvectors: ", len(TOI_EV[TOI_EV>TOI_lp]))


            ###-------------------------------------------------------------------------------------------------------###
            fig = plt.figure(figsize=(18,14))
            if showArg == 'False':
                plt.close(fig)
            if showArg == 'True':
                plt.show()

            fig.subplots_adjust(hspace=0.75,wspace=0.5)
            ax1 = fig.add_subplot(4,4,1)
            NOI_d, NOI_e = MPD(NOI_lm, NOI_lp, NOI_r)
            im1 = ax1.semilogy(NOI_d, NOI_e, color='r')
            ax1.set_title("Marchenko-Pastur Distribution \n Time Of Interest (NOI)", y=1.10, fontsize=16)
            ax1.set_xlabel("Eigenvalues", fontsize=14)
            ax1.set_ylabel("", fontsize=14)
            ax1.set_ylim(0,self.time_bins)
            ax1.set_xlim(0,3)
            ax1.grid()

            ax2 = fig.add_subplot(4,4,2)
            im2 = ax2.semilogy(NOI_pca.explained_variance_)
            ax2.hlines(NOI_lm,0,self.args.N_pop, label="lambda = lm", color='r')
            ax2.hlines(NOI_lp,0,self.args.N_pop, label="lambda = lp", color='r')
            ax2.set_title("Time Of Interest (NOI) ", y=1.10, fontsize=16)
            ax2.set_xlabel("Eigenvectors of E voltage", fontsize=14)
            ax2.set_ylabel("Eigenvalues", fontsize=14)
            ax2.grid()

            ax3 = fig.add_subplot(4,4,3)
            im3 = ax3.semilogy(TOI_pca.explained_variance_)
            ax3.hlines(TOI_lm,0,self.args.N_pop, label="lambda = lm", color="r")
            ax3.hlines(TOI_lp,0,self.args.N_pop, label="lambda = lp",color="r")
            ax3.set_title("Neuron Of Interest (TOI) ", y=1.10, fontsize=16)
            ax3.set_xlabel("Eigenvectors of E voltage", fontsize=14)
            ax3.set_ylabel("Eigenvalues", fontsize=14)
            ax3.set_ylim(0,self.time_bins)
            ax3.grid()

            ax4 = fig.add_subplot(4,4,4)
            TOI_d, TOI_e = MPD(TOI_lm, TOI_lp, TOI_r)
            im4 = ax4.semilogy(TOI_d, TOI_e, color='r')
            ax4.set_title("Marchenko-Pastur Distribution \n Time Of Interest (TOI)", y=1.10, fontsize=16)
            ax4.set_xlabel("Eigenvalues", fontsize=14)
            ax4.set_ylabel("", fontsize=14)
            ax4.set_ylim(0,self.time_bins)
            ax4.set_xlim(0,3)
            ax4.grid()

            fig.savefig(os.path.join(self.figpath, 'crossclustering_4.pdf'))


            #sns.jointplot(TOI_eig[0],TOI_eig[1])

            #merge the NOI eig 900 x component filter by whishart values
            #plt.matshow(TOI_eig[0])

            #merge the TOI eig 1001 x component filter by whishart values
            #plt.matshow(TOI_eig[0])

            #plot NOI 30X30

            ###-------------------------------------------------------------------------------------------------------------###
            # TOI EIGENVECTORS
            fig=plt.figure()
            if showArg == 'False':
                plt.close(fig)
            if showArg == 'True':
                plt.show()

            im1=plt.matshow(TOI_eig)
            plt.xlabel("[ms]", fontsize=14)
            plt.ylabel("Eigenvectors", fontsize=14)
            plt.title("TOI eigenvectors", fontsize=16)

            ###-------------------------------------------------------------------------------------------------------------###
            # NOI Eigenvectors
            fig, axs = plt.subplots(25,2, figsize=(10,dim_comps))
            #fig.suptitle("NOI Eigenvectors")
            fig.subplots_adjust(hspace = 1, wspace=0.5)
            axs = axs.ravel()


            for i in range(50): #range(len(NOI_EV[NOI_EV>TOI_lp])):
                axs[i].matshow(NOI_eig[i].reshape(self.dim_shape,self.dim_shape))
                axs[i].set_title("NOI Eigenvector " + str(i))
                axs[i].axis('off')

            fig.savefig(os.path.join(self.figpath, 'crossclustering_5.pdf'))



    def plot_conductances(self, showArg=True):
        ####---PLOTS OF THE gsynE AND gsynI TIME COURSE
        #Excitatory layer
        fig = plt.figure(figsize=(12,14))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        ax1 = fig.add_subplot(421)
        im1=ax1.imshow(self.E_gsynE)
        ax1.set_title("gsynE [uS]", fontsize=14, y=1.05)
        ax1.set_xlabel("[ms]", fontsize=14)
        ax1.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im1)

        ax2 = fig.add_subplot(422)
        im2 = ax2.imshow(self.E_gsynI)
        ax2.set_title("gsynI [uS]", fontsize=14, y=1.05)
        ax2.set_xlabel("[ms]", fontsize=14)
        ax2.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im2)

        ax = fig.add_subplot(423)
        E_gsynE_varV = []
        for i in range(len(self.E_v)):
            E_gsynE_varV.append([np.var(self.E_gsynE[i])])
        a=np.asarray(E_gsynE_varV)

        im=ax.matshow( a.reshape(self.dim_shape, self.dim_shape)); plt.colorbar(im)
        ax.set_title("gsynE Variance", y=1.05, fontsize=12)
        ax.set_xlabel("neurons", fontsize=16); ax.set_xticks([])
        ax.set_ylabel("neurons", fontsize=16)

        ax = fig.add_subplot(424)
        E_gsynE_varV = []
        for i in range(len(self.E_v)):
            E_gsynE_varV.append([np.var(self.E_gsynE[i])])
        a=np.asarray(E_gsynE_varV)
        im=ax.matshow( a.reshape(self.dim_shape, self.dim_shape)); plt.colorbar(im)
        ax.set_title("gsynI Variance", y=1.05, fontsize=12)
        ax.set_xlabel("neurons", fontsize=16); ax.set_xticks([])
        ax.set_ylabel("neurons", fontsize=16)

        ax3 = fig.add_subplot(425)
        im3 = ax3.imshow(np.cov(self.E_gsynE))
        ax3.set_title("Covariance Matrix of gsynE", fontsize=14, y=1.05)
        ax3.set_xlabel("neurons", fontsize=14)
        ax3.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im3)

        ax4 = fig.add_subplot(426)
        im4 = ax4.imshow(np.cov(self.E_gsynI))
        ax4.set_title("Covariance Matrix of gsynI", fontsize=14, y=1.05)
        ax4.set_xlabel("neurons", fontsize=14)
        ax4.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im4)

        ax5 = fig.add_subplot(427)
        im5 = ax5.imshow(np.corrcoef(self.E_gsynE))
        ax5.set_title("Correlation Matrix of gsynE", fontsize=14, y=1.05)
        ax5.set_xlabel("neurons", fontsize=14)
        ax5.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im5)

        ax6 = fig.add_subplot(428)
        im6 = ax6.imshow(np.corrcoef(self.E_gsynI))
        ax6.set_title("Correlation Matrix of gsynI", fontsize=14, y=1.05)
        ax6.set_xlabel("neurons", fontsize=14)
        ax6.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im6)

        fig.suptitle("Description of the Conductances in the E layer", fontsize=18)
        fig.subplots_adjust(hspace=0.5,wspace=0.4)

        fig.savefig(os.path.join(self.figpath, 'conductances_E_layer.pdf'))



        ####---PLOTS OF THE gsynE AND gsynI TIME COURSE
        #inhibitory layer
        fig = plt.figure(figsize=(12,14))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        ax1 = fig.add_subplot(421)
        im1=ax1.imshow(self.I_gsynE)
        ax1.set_title("gsynE [uS]", fontsize=14, y=1.05)
        ax1.set_xlabel("[ms]", fontsize=14)
        ax1.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im1)

        ax2 = fig.add_subplot(422)
        im2 = ax2.imshow(self.I_gsynI)
        ax2.set_title("gsynI [uS]", fontsize=14, y=1.05)
        ax2.set_xlabel("[ms]", fontsize=14)
        ax2.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im2)

        ax = fig.add_subplot(423)
        I_gsynE_varV = []
        for i in range(len(self.I_v)):
            I_gsynE_varV.append([np.var(self.I_gsynI[i])])
        a=np.asarray(E_gsynE_varV)
        im=ax.matshow(a.reshape(self.dim_shape, self.dim_shape)); plt.colorbar(im)
        ax.set_title("gsynE Variance", y=1.05, fontsize=12)
        ax.set_xlabel("neurons", fontsize=16); ax.set_xticks([])
        ax.set_ylabel("neurons", fontsize=16)

        ax = fig.add_subplot(424)
        I_gsynE_varV = []
        for i in range(len(self.I_v)):
            I_gsynE_varV.append([np.var(self.I_gsynE[i])])
        a=np.asarray(I_gsynE_varV)
        im=ax.matshow(a.reshape(self.dim_shape, self.dim_shape)); plt.colorbar(im)
        ax.set_title("gsynI Variance", y=1.05, fontsize=12)
        ax.set_xlabel("neurons", fontsize=16); ax.set_xticks([])
        ax.set_ylabel("neurons", fontsize=16)

        ax3 = fig.add_subplot(425)
        im3 = ax3.imshow(np.cov(self.I_gsynE))
        ax3.set_title("Covariance Matrix of gsynE", fontsize=14, y=1.05)
        ax3.set_xlabel("neurons", fontsize=14)
        ax3.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im3)

        ax4 = fig.add_subplot(426)
        im4 = ax4.imshow(np.cov(self.I_gsynI))
        ax4.set_title("Covariance Matrix of gsynI", fontsize=14, y=1.05)
        ax4.set_xlabel("neurons", fontsize=14)
        ax4.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im4)

        ax5 = fig.add_subplot(427)
        im5 = ax5.imshow(np.corrcoef(self.I_gsynE))
        ax5.set_title("Correlation Matrix of gsynE", fontsize=14, y=1.05)
        ax5.set_xlabel("neurons", fontsize=14)
        ax5.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im5)

        ax6 = fig.add_subplot(428)
        im6 = ax6.imshow(np.corrcoef(self.I_gsynI))
        ax6.set_title("Correlation Matrix of gsynI", fontsize=14, y=1.05)
        ax6.set_xlabel("neurons", fontsize=14)
        ax6.set_ylabel("neurons", fontsize=14)
        fig.colorbar(im6)

        fig.suptitle("Description of the Conductances in the I layer", fontsize=18)
        fig.subplots_adjust(hspace=0.5,wspace=0.4)

        fig.savefig(os.path.join(self.figpath, 'conductances_I_layer.pdf'))


        # Jointed Analysis in E and I layers
        from scipy import stats

        rpE = [] # correlation and p-value
        rE = [] # correlation
        rEp = [] # p-value
        for i in range(len(self.I_gsynI)):
            rpE.append(stats.pearsonr(self.E_gsynE[i], self.E_gsynI[i]))
            rE.append(rpE[i][0])
            rEp.append(rpE[i][1])

        rpI = [] # correlation and p-value
        rI = [] # correlation
        rIp = [] # p-value
        for i in range(len(self.I_gsynI)):
            rpI.append(stats.pearsonr(self.I_gsynE[i], self.I_gsynI[i]))
            rI.append(rpI[i][0])
            rIp.append(rpI[i][1])

        #
        fig = plt.figure(figsize=(12,14))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()


        ### E layer
        #
        ax = plt.subplot(321)
        im=ax.imshow(np.asarray(rE).reshape(self.dim_shape, self.dim_shape))
        ax.set_title("E layer (squared representation)", fontsize=14, y=1.05)
        ax.set_ylabel("neurons", fontsize=14)
        ax.set_xlabel("neurons", fontsize=14)
        cbar=fig.colorbar(im); cbar.set_label('Pearson correlation', rotation=-270)

        #
        ax = plt.subplot(323)

        im = ax.plot(rE)
        #
        m_rE = [] # correlation median
        s = 20 # step to compute the median
        for i in np.arange(0,self.args.N_pop-s,s):
            m_rE.append(np.median(rE[i:i+s]))
        ax.plot(range(0,self.args.N_pop-s,s),m_rE)
        ax.plot([range(self.args.N_pop)],[np.max(m_rE)])
        #
        ax.set_title("E layer (extended representation)", fontsize=14, y=1.05)
        ax.set_ylabel("Pearson's correlation", fontsize=14)
        ax.set_xlabel("neurons", fontsize=14)
        ax.set(ylim=(-1,1)); ax.grid()
        #
        ax = plt.subplot(325)
        ax.plot(rEp, 'dk') # p-value
        #ax.set_title("p-value of the correlation", fontsize=14, y=1.05)
        ax.set_ylabel("p-value", fontsize=14)
        ax.set_xlabel("neurons", fontsize=14)
        ax.set(ylim=(0,1)); ax.grid()

        ### I layer
        #
        ax = plt.subplot(322)
        im=ax.imshow(np.asarray(rI).reshape(self.dim_shape, self.dim_shape))
        ax.set_title("I layer (squared representation)", fontsize=14, y=1.05)
        ax.set_ylabel("neurons", fontsize=14)
        ax.set_xlabel("neurons", fontsize=14)
        cbar=fig.colorbar(im); cbar.set_label('Pearson correlation', rotation=-270)

        #
        ax = plt.subplot(324)
        ax.plot(rI)
        #
        m_rI = [] # correlation median
        s = 20 # step to compute the median
        for i in np.arange(0,self.args.N_pop-s,s):
            m_rI.append(np.median(rI[i:i+s]))
        ax.plot(range(0,self.args.N_pop-s,s),m_rI)
        ax.plot([range(self.args.N_pop)],[np.max(m_rI)])
        #
        ax.set_title("I layer (extended representation)", fontsize=14, y=1.05)
        ax.set_ylabel("Pearson's correlation", fontsize=14)
        ax.set_xlabel("neurons", fontsize=14)
        ax.set(ylim=(-1,1)); ax.grid()
        #
        ax = plt.subplot(326)
        ax.plot(rIp, 'dk') # p-value
        #ax4.set_title("p-value of the correlation", fontsize=14, y=1.05)
        ax.set_ylabel("p-value", fontsize=14)
        ax.set_xlabel("neurons", fontsize=14)
        ax.set(ylim=(0,1)); ax.grid()

        fig.suptitle("Correlation between conductances (gsynE and gsynI)", fontsize=18)
        fig.subplots_adjust(hspace=0.5,wspace=0.4)
        fig.savefig(os.path.join(self.figpath, 'conductances_JointAnalysis.pdf'))



    def plot_signalprocessing(self, S_fr, E_fr, I_fr, showArg=True):
        ###---SIGNAL PROCESSING AND REPRESENTATION (amplitude, frequency, phase)
        # TO DO
        # note that changes in signals' phase indicate travelling waves
        # https://www.nature.com/articles/nrn.2018.20
        # desinchronization as traveling waves indicator

        from scipy import signal
        from scipy.signal import hilbert as hht

        data = [self.E_v, self.E_gsynE, self.E_gsynI,
                self.I_v, self.I_gsynE, self.I_gsynI,
                S_fr, E_fr, I_fr]

        analytic_signals = []
        inst_phases = []
        for i in range(len(data)):
            analytic_signals.append(hht(data[i]))
            inst_phases.append(np.unwrap(np.angle(hht(data[i]))))

        # pcolor plots of recorded variables
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows,cols,figsize=(11,7))
        fig.tight_layout(pad=0.0)
        fig.suptitle('Phase variations on Recorded Variables', y=1.10, fontsize=16)
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        axes_list = fig.axes
        titles_list = ['E_v','E_gsynE','E_gsynI','I_v','I_gsynE','I_gsynI', 'S_fr','E_fr','I_fr']
        for i in range(rows*cols):
            im = axes_list[i].matshow(inst_phases[i]); fig.colorbar(im,ax=axes_list[i])
            axes_list[i].set_ylim(0, self.args.N_pop)
            axes_list[i].set_title(titles_list[i], fontsize=14, y=1)
            axes_list[i].set_xlabel('[ms]')
            axes_list[i].set_ylabel('neurons')
        fig.subplots_adjust(hspace=0.3,wspace=0.4)
        fig.savefig(os.path.join(self.figpath, 'signalProcessing_1.pdf'))


        ## insert fft and spectrum
        # pcolor plots of spike firing rates
        rows = 1
        cols = 3
        fig, axes = plt.subplots(rows,cols,figsize=(11,3))
        fig.tight_layout(pad=0.0)
        fig.suptitle('Phase variations on Spikes Firing Rates', y=1.30, fontsize=16)
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        axes_list = fig.axes
        titles_list = ['S_fr','E_fr','I_fr']
        for i in range(rows*cols):
            im2 = axes_list[i].matshow(inst_phases[6+i]); fig.colorbar(im2,ax=axes_list[i])
            axes_list[i].set_ylim(0, self.args.N_pop)
            axes_list[i].set_title(titles_list[i], fontsize=14, y=1)
            axes_list[i].set_xlabel('[ms]')
            axes_list[i].set_ylabel('neurons')
        fig.subplots_adjust(hspace=0.3,wspace=0.4)

        fig.savefig(os.path.join(self.figpath, 'signalProcessing_2.pdf'))

    def plot_movingbumps(self):

        def multisave_plot_surface(d, step, path, layer):

            if str(d.name)  == 'v':
                units = 'mV'
            else:
                units = 'uS'

            d_name =  str(d.name)
            title = layer + ' | ' + d_name + ' [' + units + ']'

            d = d.as_array().reshape(int(np.sqrt(len(d))),int(np.sqrt(len(d))),int(len(d[0])))

            for i in np.arange(10, len(d[0][0]), step):

                X = np.arange(0, len(d), 1)
                X, Y = np.meshgrid(X, X)
                Z = d[:,:,i]
                #
                plt.clf()
                fig = plt.figure(1, figsize = (5,5))

                #ax = fig.gca(projection='3d')
                ax = fig.gca()

                #im = ax.plot_surface(X, Y, Z,
                #            rstride = 1,
                #            cstride = 1,
                #            alpha = 0.85,
                #            cmap = 'coolwarm',
                #            linewidth = 0,
                #            antialiased = True,
                #            edgecolor = 'none')

                im = ax.pcolormesh(X, Y, Z)

                #cset = ax.contourf(X, Y, Z, zdir='z', offset = min(d[0].ravel())-20, cmap = 'coolwarm')
                #cset = ax.contourf(X, Y, Z, zdir='y', offset = max(X.ravel())+10, cmap = 'coolwarm')
                #cset = ax.contourf(X, Y, Z, zdir='x', offset = min(X.ravel())-10, cmap = 'coolwarm')
                #cset = ax.contourf(X, Y, Z, zdir='z', offset = -300, cmap = 'coolwarm')

                cb = fig.colorbar(im, fraction = 0.01, pad = 0.1)
                #.set_clim(-75,0)
                cb.set_label('', rotation=0)
                       #
                ax.set_title(title + ' | ' + str(i) + ' [ms]')
                ax.set_xlabel('neurons')
                ax.set_xlim(min(X.ravel()), max(X.ravel()))
                ax.set_ylabel('neurons')
                ax.set_ylim(min(X.ravel()), max(X.ravel()))
                #ax.set_zlabel('')
                #ax.set_zlim(min(d[0].ravel()), max(d[0].ravel())
                #ax.set_zlim(-100,20)
                #ax.view_init(elev=90, azim=90)


                fig.tight_layout()

                plt.savefig(path + layer + '/' + d_name + "_{0:04}.png".format(i), dpi=150)

            import subprocess
            import os
            i = path + layer + '/' + d_name + '*.png'
            o = path + layer + '/' + d_name + '.gif'
            subprocess.call('convert -delay 100 -loop 0 ' + i + ' ' + o, shell = True)
            subprocess.call('rm ' + path + layer + '/' + d_name + '*.png', shell = True)


        data = [self.E_v,
                self.E_gsynE,
                self.E_gsynI,
                self.I_v,
                self.I_gsynE,
                self.I_gsynI]

        path = os.path.join(self.figpath + "/moving_bump/")

        step = 10
        for i in range(len(data)):
            print(i)
            if i <= 2:
                layer = 'E_layer'
            else:
                layer = 'I_layer'

            multisave_plot_surface(data[i], step, path, layer)

        from IPython.display import Image
        from IPython.display import display
        import matplotlib.image as mpimg
        #ge = Image(filename='./moving_bump_png/E_layer/gsyn_exc.gif')
        #gi = Image(filename='./moving_bump_png/E_layer/gsyn_inh.gif')
        v = Image(filename='./moving_bump_png/E_layer/v.gif')
        Fplotdisplay(v)
        #ge = Image(filename='./moving_bump_png/I_layer/gsyn_exc.gif')
        #gi = Image(filename='./moving_bump_png/I_layer/gsyn_inh.gif')
        v = Image(filename='./moving_bump_png/I_layer/v.gif')
        display(v)

    def plot_projections(self, showArg=True):
        ###---Probabilistic connections check
        import seaborn as sns
        bins = len(self.E_v)

        # print(len(self.projEE[:]))
        # print(len(self.projEI[:]))
        # print(len(self.projEIone[:]))
        # print(len(self.projII[:]))
        # print(len(self.projIE[:]))
        # print(len(self.projIEone[:]))

        fig = plt.figure(figsize=(6,12),constrained_layout=True)
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()
        fig.subplots_adjust(hspace=0.5,wspace=0.8)


        ax1 = fig.add_subplot(6,2,1)
        sns.distplot(self.projEE.T[0], kde=False, bins=bins)
        ax1.set_title("Projections EE", fontsize=14,y=1.05)
        #ax1.set_ylim(0, 60)

        ax2 = fig.add_subplot(6,2,2)
        adM = np.zeros((bins,bins))
        for i in np.arange(len(self.projEE.T[0])):
            for j in np.arange(bins):
                if self.projEE.T[1][i] == j:
                    x = round(int(self.projEE.T[1][i]))
                    y = round(int(self.projEE.T[0][i]))
                    adM[x][y] = adM[x][y] + 1
        im2 = ax2.matshow(adM)
        ax2.set_title("Connectivity Matrix EE", fontsize=14, y=1.10)
        ax2.set_xlabel("E layer", fontsize=14);
        ax2.set_ylabel("E layer", fontsize=14);
        ax2.set_xticks([])
        ax2.set_yticks([])

        #ax3 = fig.add_subplot(6,2,3)
        #sns.distplot(self.projEI.T[0], kde=False, bins=bins)
        #ax3.set_title("Projections EI", fontsize=14,y=1.05)
        #ax3.set_ylim(0, 60)

        #ax4 = fig.add_subplot(6,2,4)
        #adM = np.zeros((bins,bins))
        #for i in np.arange(len(self.projEI.T[0])):
        #    for j in np.arange(bins):
        #        if self.projEI.T[1][i] == j:
        #            x = round(int(self.projEI.T[1][i]))
        #            y = round(int(self.projEI.T[0][i]))
        #            adM[x][y] = adM[x][y] + 1
        #im4 = ax4.matshow(adM)
        #ax4.set_title("Connectivity Matrix EI", fontsize=14, y=1.10)
        #ax4.set_xlabel("E layer", fontsize=14);
        #ax4.set_ylabel("I layer", fontsize=14);
        #ax4.set_xticks([])
        #ax4.set_yticks([])

        ax5 = fig.add_subplot(6,2,5)
        sns.distplot(self.projEIone.T[0], kde=False, bins=bins)
        ax5.set_title("Projections EIone", fontsize=14,y=1.05)
        ax5.set_ylim(0, 5)

        ax6 = fig.add_subplot(6,2,6)
        adM = np.zeros((bins,bins))
        for i in np.arange(len(self.projEIone.T[0])):
            for j in np.arange(bins):
                if self.projEIone.T[1][i] == j:
                    x = round(int(self.projEIone.T[1][i]))
                    y = round(int(self.projEIone.T[0][i]))
                    adM[x][y] = adM[x][y] + 1
        im6 = ax6.matshow(adM)
        ax6.set_title("Connectivity Matrix IEone", fontsize=14, y=1.10)
        ax6.set_xlabel("I layer", fontsize=14);
        ax6.set_ylabel("E layer", fontsize=14);
        ax6.set_xticks([])
        ax6.set_yticks([])

        ax7 = fig.add_subplot(6,2,7)
        sns.distplot(self.projII.T[0], kde=False, bins=bins)
        ax7.set_title("Projections II", fontsize=14,y=1.05)
        #ax7.set_ylim(0, 60)

        ax8 = fig.add_subplot(6,2,8)
        adM = np.zeros((bins,bins))
        for i in np.arange(len(self.projII.T[0])):
            for j in np.arange(bins):
                if self.projII.T[1][i] == j:
                    x = round(int(self.projII.T[1][i]))
                    y = round(int(self.projII.T[0][i]))
                    adM[x][y] = adM[x][y] + 1
        im8 = ax8.matshow(adM)
        ax8.set_title("Connectivity Matrix II", fontsize=14, y=1.10)
        ax8.set_xlabel("I layer", fontsize=14);
        ax8.set_ylabel("I layer", fontsize=14);
        ax8.set_xticks([])
        ax8.set_yticks([])

        #ax9 = fig.add_subplot(6,2,9)
        #a, _ = np.histogram(self.projIE.T[0], bins=np.unique(self.projIE.T[0]))
        #sns.distplot(self.projIE.T[0], kde=False, bins=self.args.N_pop)
        #ax9.set_title("Projections IE", fontsize=14,y=1.05)
        #ax9.set_ylim(0, 60)

        #ax10 = fig.add_subplot(6,2,10)
        #adM = np.zeros((bins,bins))
        #for i in np.arange(len(self.projIE.T[0])):
        #    for j in np.arange(bins):
        #        if self.projIE.T[1][i] == j:
        #            x = round(int(self.projIE.T[1][i]))
        #            y = round(int(self.projIE.T[0][i]))
        #            adM[x][y] = adM[x][y] + 1
        #im10 = ax10.matshow(adM)
        #ax10.set_title("Connectivity Matrix IE", fontsize=14, y=1.10)
        #ax10.set_xlabel("I layer", fontsize=14);
        #ax10.set_ylabel("E layer", fontsize=14);
        #ax10.set_xticks([])
        #ax10.set_yticks([])

        ax11 = fig.add_subplot(6,2,11)
        sns.distplot(self.projIEone.T[0], kde=False, bins=bins)
        ax11.set_title("Projections IEone", fontsize=14,y=1.05)
        ax11.set_ylim(0, 5)
        fig.subplots_adjust(hspace=0.6,wspace=1)

        ax12 = fig.add_subplot(6,2,12)
        adM = np.zeros((bins,bins))
        for i in np.arange(len(self.projIEone.T[0])):
            for j in np.arange(bins):
                if self.projIEone.T[1][i] == j:
                    x = round(int(self.projIEone.T[1][i]))
                    y = round(int(self.projIEone.T[0][i]))
                    adM[x][y] = adM[x][y] + 1
        im12 = ax12.matshow(adM)
        ax12.set_title("Connectivity Matrix IEone", fontsize=14, y=1.10)
        ax12.set_xlabel("I layer", fontsize=14);
        ax12.set_ylabel("E layer", fontsize=14);
        ax12.set_xticks([])
        ax12.set_yticks([])

        fig.savefig(os.path.join(self.figpath, 'projections.png'))


    def plot_overview(self, showArg=True):

        #make the plots
        fig = plt.figure(figsize=(14,12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle("(wEE = {}) - (wIE = {}) - (dE = exp(-d**(1/{})) - (dI = exp(-d**(1/{})))".format(self.args.E_weight, self.args.I_weight, self.args.E_expPar, self.args.I_expPar), fontsize=16)
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        ax1 = fig.add_subplot(231)
        im1=ax1.matshow(self.E_v);
        ax1.set_title("voltage in E layer", fontsize=14, y=1.15)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2.5%", pad=0.10)
        fig.colorbar(im1, cax=cax)

        ax2 = fig.add_subplot(234)
        im2=ax2.matshow(self.I_v);
        ax2.set_title("voltage in I layer", fontsize=14, y=1.15)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="2.5%", pad=0.10)
        fig.colorbar(im2, cax=cax)

        E_varV = []
        for i in range(len(self.E_v)):
            E_varV.append([np.var(self.E_v[i])])
        a=np.asarray(E_varV)
        a.resize(int(np.sqrt(len(self.E_v))),int(np.sqrt(len(self.E_v))))
        ax3 = fig.add_subplot(232)
        im3=ax3.matshow(a);
        ax3.set_title("voltage variance in E layer", fontsize=14, y=1.15)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="2.5%", pad=0.10)
        fig.colorbar(im3, cax=cax)

        I_varV = []
        for i in range(len(self.I_v)):
            I_varV.append([np.var(self.I_v[i])])
        b=np.asarray(I_varV)
        b.resize(int(np.sqrt(len(self.I_v))),int(np.sqrt(len(self.I_v))))
        ax4 = fig.add_subplot(235)
        im4=ax4.matshow(b);
        ax4.set_title("voltage variance in I layer", fontsize=14, y=1.15)
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="2.5%", pad=0.10)
        fig.colorbar(im4, cax=cax)

        x = np.arange(self.args.N_pop)
        ax5 = fig.add_subplot(233)
        ax5.semilogx(np.exp(-x**(1/self.args.E_expPar)))
        #ax5.semilogx(1/self.args.E_expPar * np.exp(-x/self.args.E_expPar)) #see https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        ax5.set_title("dE = exp(-d**(1/{})".format(self.args.E_expPar), fontsize=14, y=1.10)
        #ax5.set_title("dE = 1/{} * exp(-d/{})".format(self.args.E_expPar, self.args.I_expPar), fontsize=14, y=1.10)
        ax5.set_xlabel("Neurons")
        ax5.set_ylabel("Probability of make a synapse")
        ax5.set_ylim([0.0, 1.0])
        ax5.grid(True)

        x = np.arange(self.args.N_pop)
        ax6 = fig.add_subplot(236)
        ax6.semilogx(np.exp(-x**(1/self.args.I_expPar)))
        #ax6.semilogx(1/self.args.I_expPar * np.exp(-x/self.args.I_expPar)) #see https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        ax6.set_title("dI = exp(-d**(1/{})".format(self.args.I_expPar), fontsize=14, y=1.10)
        ax6.set_xlabel("Neurons")
        ax6.set_ylabel("Probability of make a synapse")
        ax6.set_ylim([0.0,1.0])
        ax6.grid(True)

        fig.savefig(os.path.join(self.figpath, 'basic_plot.pdf'))
        #fig.savefig("../outputs/input_2D/prob/voltage_{}_{}_{}_{}.png".format(self.args.exw, self.args.inw, self.args.E_root, self.args.I_root))

    def plot_movingWaves(self): #not working

        def multisave_plot_surface(d, step, path, layer):

            d = d.as_array().reshape(int(np.sqrt(len(d))),int(np.sqrt(len(d))),int(len(d[0])))


            title = layer + ' | ' + 'phase' + '[angle]'
            for i in np.arange(10, len(d[0][0]), step):

                X = np.arange(0, len(d), 1)
                X, Y = np.meshgrid(X, X)
                Z = d[:,:,i]
                #
                plt.clf()
                fig = plt.figure(1, figsize = (5,5))

                #ax = fig.gca(projection='3d')
                ax = fig.gca()

                #im = ax.plot_surface(X, Y, Z,
                #            rstride = 1,
                #            cstride = 1,
                #            alpha = 0.85,
                #            cmap = 'coolwarm',
                #            linewidth = 0,
                #            antialiased = True,
                #            edgecolor = 'none')

                im = ax.pcolormesh(X, Y, Z)

                #cset = ax.contourf(X, Y, Z, zdir='z', offset = min(d[0].ravel())-20, cmap = 'coolwarm')
                #cset = ax.contourf(X, Y, Z, zdir='y', offset = max(X.ravel())+10, cmap = 'coolwarm')
                #cset = ax.contourf(X, Y, Z, zdir='x', offset = min(X.ravel())-10, cmap = 'coolwarm')
                #cset = ax.contourf(X, Y, Z, zdir='z', offset = -300, cmap = 'coolwarm')

                cb = fig.colorbar(im, fraction = 0.01, pad = 0.1)
                cb.set_clim(-360,360)
                cb.set_label('', rotation=0)
                       #
                ax.set_title(title + ' | ' + str(i) + ' [ms]')
                ax.set_xlabel('neurons')
                ax.set_xlim(min(X.ravel()), max(X.ravel()))
                ax.set_ylabel('neurons')
                ax.set_ylim(min(X.ravel()), max(X.ravel()))
                #ax.set_zlabel('')
                #ax.set_zlim(min(d[0].ravel()), max(d[0].ravel())
                #ax.set_zlim(-100,20)
                #ax.view_init(elev=90, azim=90)


                fig.tight_layout()

                plt.savefig(path + layer + '/' + 'phase' + "_{0:04}.png".format(i), dpi=150)

            import subprocess
            import os
            i = path + layer + '/' + 'phase' + '*.png'
            o = path + layer + '/' + 'phase' + '.gif'
            subprocess.call('convert -delay 100 -loop 0 ' + i + ' ' + o, shell = True)
            subprocess.call('rm ' + path + layer + '/' + 'phase' + '*.png', shell = True)


        data = [self.E_v,
                self.I_v]

        path = os.path.join(self.figpath + "/moving_bump/")

        step = 10
        for i in range(len(data)):
            print(i)
            if i <= 0:
                layer = 'E_layer'
            else:
                layer = 'I_layer'

            multisave_plot_surface(data[i], step, path, layer)

        return data

    def plot_abstract_old(self, E_fr, I_fr, showArg=True):
        # pcolor plots
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows,cols,figsize=(20,10))
        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        fig.tight_layout(pad=3.0)
        axes_list = fig.axes
        titles_list = [   'E var 0:249 [ms]',
                          'E var 250:499 [ms]',
                          'E var 500:749 [ms]',
                          'E var 750:999 [ms]',
                          'I var 0:249 [ms]',
                          'I var 250:499 [ms]',
                          'I var 500:749 [ms]',
                          'I var 750:999 [ms]']


        for i in range(rows*cols-8):
            if i <= 3:
                E_varV = []
                if i == 0:
                    for j in range(len(self.E_v)):
                        E_varV.append([np.var(self.E_v[j][0:250])])

                if i == 1:
                    for j in range(len(self.E_v)):
                        E_varV.append([np.var(self.E_v[j][250:500])])

                if i == 2:
                    for j in range(len(self.E_v)):
                        E_varV.append([np.var(self.E_v[j][500:750])])

                if i == 3:
                    for j in range(len(self.E_v)):
                        E_varV.append([np.var(self.E_v[j][750:1000])])


                E_varV=np.asarray(E_varV)
                E_varV.resize(int(np.sqrt(len(self.E_v))),int(np.sqrt(len(self.E_v))))
                im1=axes_list[i].matshow(E_varV, cmap='Reds')
                #fig.colorbar(axes_list[i].matshow(E_varV))
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=12)
                #axes_list[i].set_xlabel("neurons", fontsize=10); axes_list[i].set_xticks([])
                #axes_list[i].set_ylabel("neurons", fontsize=10)

            if i >= 4:
                I_varV = []
                if i == 4:
                    for j in range(len(self.I_v)):
                        I_varV.append([np.var(self.I_v[j][0:250])])

                if i == 5:
                    for j in range(len(self.I_v)):
                        I_varV.append([np.var(self.I_v[j][250:500])])

                if i == 6:
                    for j in range(len(self.I_v)):
                        I_varV.append([np.var(self.I_v[j][500:750])])

                if i == 7:
                    for j in range(len(self.I_v)):
                        I_varV.append([np.var(self.I_v[j][750:1000])])


                I_varV=np.asarray(I_varV)
                I_varV.resize(int(np.sqrt(len(self.I_v))),int(np.sqrt(len(self.I_v))))
                im2=axes_list[i].matshow(I_varV,  cmap='Blues')
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=12)
                #axes_list[i].set_xlabel("neurons", fontsize=10); axes_list[i].set_xticks([])
                #axes_list[i].set_ylabel("neurons", fontsize=10)


            # input representations
            from VisionSNN.dot_input import generate_dot
            ###---LOADING INPUT -------------------------------------------------------------------------------
            self.time_bins = int(self.args.simtime/self.args.dt) #extended time
            #extended pixels
            sqrt_pop_size = np.sqrt(self.args.N_pop)
            width = int(sqrt_pop_size)
            #1.1 generate 2d with time stimulus
            inputBar = generate_dot(width, width,
                                    self.time_bins,
                                    dot_size=0.1,
                                    im_noise=self.args.im_noise,
                                    im_contrast=self.args.im_contrast,
                                    )
            inputBar = inputBar.transpose(2, 1, 0)

            inputBarVar = []
            for k in range(self.args.N_pop):
                    inputBarVar.append(inputBar.reshape(1000,self.args.N_pop)[0:250,k].var())

            axes_list[8].matshow(np.asarray(inputBarVar).reshape(30,30), cmap='binary')
            axes_list[8].set_title('Input var 0:249 [ms]', y=1.10, fontsize=12)
            axes_list[8].set_xlabel("", fontsize=10); axes_list[i].set_xticks([])
            axes_list[8].set_ylabel("", fontsize=10)

            inputBarVar = []
            for k in range(self.args.N_pop):
                    inputBarVar.append(inputBar.reshape(1000,self.args.N_pop)[250:500,k].var())

            axes_list[9].matshow(np.asarray(inputBarVar).reshape(30,30), cmap='binary')
            axes_list[9].set_title('Input var 0:249 [ms]', y=1.10, fontsize=12)
            axes_list[9].set_xlabel("", fontsize=10); axes_list[i].set_xticks([])
            axes_list[9].set_ylabel("", fontsize=10)

            inputBarVar = []
            for k in range(self.args.N_pop):
                    inputBarVar.append(inputBar.reshape(1000,self.args.N_pop)[500:750,k].var())

            axes_list[10].matshow(np.asarray(inputBarVar).reshape(30,30), cmap='binary')
            axes_list[10].set_title('Input var 500:750 [ms]', y=1.10, fontsize=12)
            axes_list[10].set_xlabel("", fontsize=10); axes_list[i].set_xticks([])
            axes_list[10].set_ylabel("", fontsize=10)

            inputBarVar = []

            for k in range(self.args.N_pop):
                    inputBarVar.append(inputBar.reshape(1000,self.args.N_pop)[750:1000,k].var())

            axes_list[11].matshow(np.asarray(inputBarVar).reshape(30,30), cmap='binary')
            axes_list[11].set_title('Input var 750:1000 [ms]', y=1.10, fontsize=12)
            axes_list[11].set_xlabel("", fontsize=10); axes_list[i].set_xticks([])
            axes_list[11].set_ylabel("", fontsize=10)


            # firing rate
            axes_list[8].matshow(np.asarray(inputBarVar).reshape(30,30), cmap='binary')
            axes_list[8].set_title('Input var 0:249 [ms]', y=1.10, fontsize=12)
            axes_list[8].set_xlabel("", fontsize=10); axes_list[i].set_xticks([])
            axes_list[8].set_ylabel("", fontsize=10)

            inputBarVar = []
            for k in range(self.args.N_pop):
                    inputBarVar.append(inputBar.reshape(1000,900)[250:500,k].var())

            axes_list[9].matshow(np.asarray(inputBarVar).reshape(30,30), cmap='binary')
            axes_list[9].set_title('Input var 0:249 [ms]', y=1.10, fontsize=12)
            axes_list[9].set_xlabel("", fontsize=10); axes_list[i].set_xticks([])
            axes_list[9].set_ylabel("", fontsize=10)

            inputBarVar = []
            for k in range(900):
                    inputBarVar.append(inputBar.reshape(1000,900)[500:750,k].var())
            axes_list[10].matshow(np.asarray(inputBarVar).reshape(30,30), cmap='binary')
            axes_list[10].set_title('Input var 0:249 [ms]', y=1.10, fontsize=12)
            axes_list[10].set_xlabel("", fontsize=10); axes_list[i].set_xticks([])
            axes_list[10].set_ylabel("", fontsize=10)

            inputBarVar = []
            for k in range(900):
                    inputBarVar.append(inputBar.reshape(1000,900)[750:1000,k].var())
            axes_list[11].matshow(np.asarray(inputBarVar).reshape(30,30), cmap='binary')
            axes_list[11].set_title('Input var 0:249 [ms]', y=1.10, fontsize=12)
            axes_list[11].set_xlabel("", fontsize=10); axes_list[i].set_xticks([])
            axes_list[11].set_ylabel("", fontsize=10)


            # 1
            E_fr_min = []
            E_fr_max = []
            I_fr_min = []
            I_fr_max = []
            for m in range(250):
                E_fr_min.append(E_fr[:,m].min())
                E_fr_max.append(E_fr[:,m].max())
                I_fr_min.append(I_fr[:,m].min())
                I_fr_max.append(I_fr[:,m].max())

            x = np.arange(250)
            Ey1 = E_fr_min
            Ey2 = E_fr_max
            Iy1 = I_fr_min
            Iy2 = I_fr_max

            axes_list[12].fill_between(x, Ey1, Ey2,
                             facecolor="red", # The fill color
                             color='red',       # The outline color
                             alpha=0.5)
            axes_list[12].set_title('Firing Rates', y=1.10, fontsize=12)
            axes_list[12].set_xlabel("[ms]", fontsize=10); axes_list[i].set_xticks([])
            axes_list[12].set_ylabel("[Hz]", fontsize=10)
            axes_list[12].fill_between(x, Iy1, Iy2,
                             facecolor="blue", # The fill color
                             color='blue',       # The outline color
                             alpha=0.5)

            # 2
            E_fr_min = []
            E_fr_max = []
            I_fr_min = []
            I_fr_max = []
            for i in range(250,500):
                E_fr_min.append(E_fr[:,i].min())
                E_fr_max.append(E_fr[:,i].max())
                I_fr_min.append(I_fr[:,i].min())
                I_fr_max.append(I_fr[:,i].max())

            x = np.arange(250,500)
            Ey1 = E_fr_min
            Ey2 = E_fr_max
            Iy1 = I_fr_min
            Iy2 = I_fr_max

            axes_list[13].fill_between(x, Ey1, Ey2,
                             facecolor="red", # The fill color
                             color='red',       # The outline color
                             alpha=0.5)
            axes_list[13].set_title('Firing Rates', y=1.10, fontsize=12)
            axes_list[13].set_xlabel("[ms]", fontsize=10);
            #axes_list[i].set_xticks([])
            axes_list[13].set_ylabel("[Hz]", fontsize=10)
            axes_list[13].fill_between(x, Iy1, Iy2,
                             facecolor="blue", # The fill color
                             color='blue',       # The outline color
                             alpha=0.5)

            # 3
            E_fr_min = []
            E_fr_max = []
            I_fr_min = []
            I_fr_max = []
            for i in range(500,750):
                E_fr_min.append(E_fr[:,i].min())
                E_fr_max.append(E_fr[:,i].max())
                I_fr_min.append(I_fr[:,i].min())
                I_fr_max.append(I_fr[:,i].max())

            x = np.arange(500,750)
            Ey1 = E_fr_min
            Ey2 = E_fr_max
            Iy1 = I_fr_min
            Iy2 = I_fr_max

            axes_list[14].fill_between(x, Ey1, Ey2,
                             facecolor="red", # The fill color
                             color='red',       # The outline color
                             alpha=0.5)
            axes_list[14].set_title('Firing Rates', y=1.10, fontsize=12)
            axes_list[14].set_xlabel("[ms]", fontsize=10);
            #axes_list[i].set_xticks([])
            axes_list[14].set_ylabel("[Hz]", fontsize=10)
            axes_list[14].fill_between(x, Iy1, Iy2,
                             facecolor="blue", # The fill color
                             color='blue',       # The outline color
                             alpha=0.5)

            # 4
            E_fr_min = []
            E_fr_max = []
            I_fr_min = []
            I_fr_max = []
            for p in range(750,1000):
                E_fr_min.append(E_fr[:,p].min())
                E_fr_max.append(E_fr[:,p].max())
                I_fr_min.append(I_fr[:,p].min())
                I_fr_max.append(I_fr[:,p].max())

            x = np.arange(750,1000)
            Ey1 = E_fr_min
            Ey2 = E_fr_max
            Iy1 = I_fr_min
            Iy2 = I_fr_max

            axes_list[15].fill_between(x, Ey1, Ey2,
                             facecolor="red", # The fill color
                             color='red',       # The outline color
                             alpha=0.5)
            axes_list[15].set_title('Firing Rates', y=1.10, fontsize=12)
            axes_list[15].set_xlabel("[ms]", fontsize=10);
            #axes_list[i].set_xticks([])
            axes_list[15].set_ylabel("[Hz]", fontsize=10)
            axes_list[15].fill_between(x, Iy1, Iy2,
                             facecolor="blue", # The fill color
                             color='blue')       # The outline color)

            fig.savefig(os.path.join(self.figpath, 'plot_figure1_old.png'))

    def plot_figure1(self, perStart, perEnd, perStep, showArg=True):

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import cm
        import numpy as np

        # TAKE THE FIRING RATE
        S_spikes = self.spikeSources.segments[0].spiketrains
        E_spikes = self.popEdata.segments[0].spiketrains
        I_spikes = self.popIdata.segments[0].spiketrains

        # FIRING RATE FUNCTION DEFINITION
        N_pop = self.args.N_pop
        time_bins = self.time_bins
        N_tau = self.args.N_tau
        dt = 1/ time_bins
        def firing_rate(b, N_tau=N_tau) :
            fr = np.zeros_like(b).astype(float)
            for t in range(b.shape[0]) :
            #print(1 - 1 / tau, t, fr.mean())
                fr[t, :] = (1 - 1 / N_tau) * fr[t-1, :] + 1 / N_tau * b[t, :]
            return fr.T # transposition to have (N,T)

        # FIRING RATES OF SOURCES SPIKES
        S_binMat = np.zeros((time_bins, N_pop), dtype='int')
        S_spk_list = []
        for i in range(N_pop): # loops over sources
            S_spike_train = S_spikes[i].as_array() # makes as array each source
            S_spike_times = [int(t) for t in S_spike_train]
            S_spk_list.append(S_spikes[i].as_array())
            for spike_time in S_spike_times: # loops over spikes for each source
                    #print(spike_time, i)
                    S_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                    # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        S_fr = firing_rate(S_binMat)*(1/dt)

        # FIRING RATES OF E LAYER SPIKES
        E_binMat = np.zeros((time_bins, N_pop), dtype='int')
        E_spk_list = []
        for i in range(N_pop): # loops over sources
            E_spike_train = E_spikes[i].as_array() # makes as array each source
            E_spk_list.append(E_spikes[i].as_array())
            E_spike_times = [int(t) for t in E_spike_train]
            for spike_time in E_spike_times: # loops over spikes for each source
                #print(spike_time, i)
                E_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        E_fr = firing_rate(E_binMat)*(1/dt)

        # FIRING RATES OF I LAYER SPIKES
        I_binMat = np.zeros((time_bins, N_pop), dtype='int')
        I_spk_list = []
        for i in range(N_pop): # loops over sources
            I_spike_train = I_spikes[i].as_array() # makes as array each source
            I_spk_list.append(I_spikes[i].as_array())
            I_spike_times = [int(t) for t in I_spike_train]
            for spike_time in I_spike_times: # loops over spikes for each source
                #print(spike_time, i)
                I_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        I_fr = firing_rate(I_binMat)*(1/dt)

        # FIRING RATES OF SOURCE, EXC AND INH LAYER
        fr = [S_fr, E_fr, I_fr]

        # MAKE THE PLOTS
        cmap = cm.PRGn
        rows = 3
        cols = 5
        fig, axes = plt.subplots(rows,cols,figsize=(15,8))
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)#(pad=0.1)#(pad=0.4)
        fig.subplots_adjust(top=0.92, bottom=0.08,
                left=0.10, #0.10,
                right=0.95, #0.95,
                hspace=0.05,
                wspace=0.35)

        def plot_figure1_gen(figIndex, fr):
            periods = np.linspace(perStart, perEnd, int(perEnd/perStep)+1)
            nets = 3 # input, exc, inh nets
            axes_list = fig.axes
            dim = int(np.sqrt(self.args.N_pop))

            if (figIndex < 5):
                i = figIndex
                fr = fr[0]
                colorMap = 'binary'
            if (figIndex < 10) & (figIndex >= 5):
                i = figIndex - 5
                fr = fr[1]
                colorMap = 'Reds'
            if (figIndex > 9):
                i = figIndex - 10
                fr = fr[2]
                colorMap = 'Blues'

            In_maxFr = []
            for j in range(self.args.N_pop):
                In_maxFr.append(fr[j, int(periods[i]):int(periods[i+1])].mean())

            im = axes_list[figIndex].matshow(np.asarray(In_maxFr).reshape(dim,dim), cmap=colorMap)
            axes_list[figIndex].set_title(titles_list[i], y=1.10, fontsize=14)
            axes_list[figIndex].set_ylabel('neurons', fontsize=14)
            #axes_list[i].set_xlabel('neurons', fontsize=14)
            axes_list[figIndex].set_xticks([])
            axes_list[figIndex].set_yticks([])
            fig.colorbar(im, ax=axes_list[figIndex],fraction=0.046, pad=0.04)

        # generates plot
        titles_list = [
                      '0:200 [ms] | [Hz]',
                      '200:400 [ms] | [Hz]',
                      '400:600 [ms] | [Hz]',
                      '600:800 [ms] | [Hz]',
                      '800:1000 [ms] | [Hz]',

                      '',
                      '',
                      '',
                      '',
                      '',

                      '',
                      '',
                      '',
                      '',
                      '']

        for tit in range(len(titles_list)):
            plot_figure1_gen(tit, fr)

        fig.savefig(os.path.join(self.figpath, 'plot_figure1.pdf'))


    def plot_figure2old(self, fr, nS, nM, nL, showArg=True):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np
        cmap = cm.PRGn
        rows = 4
        cols = 3
        fig, axes = plt.subplots(rows,cols,figsize=(10,20))
        fig.tight_layout(pad=0.4)
        axes_list = fig.axes
        fig.suptitle('Averaged firing rate of neurons in relation to specific input space location',  fontsize=20)
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                            wspace=0.35)

        def plot_meanFR(fr, neuron, subplot):
            pop = self.args.N_pop
            FR = fr
            FR_subset = []
            radius = 2
            for j in range(len(FR)):
                Ax = neuron // int(np.sqrt(pop))
                Ay = neuron % int(np.sqrt(pop))
                Bx = j // int(np.sqrt(pop))
                By = j % int(np.sqrt(pop))
                d = np.sqrt((Ax - Bx)**2 + (Ay - By)**2) #abs(i-j)
                if d <= radius:
                        FR_subset.append(j)
            FR_mean = []
            for fr in range(len(FR)):
                    FR_mean.append(np.mean(FR[FR_subset].T[fr]))
            for subset in FR_subset:
                if subset == neuron:
                    axes_list[subplot].plot(FR[subset], '+:r', label='FR of the {}th cell'.format(neuron))
                else:
                    axes_list[subplot].plot(FR[subset], '+:y')

            #axes_list[subplot].plot(FR_mean, '+:k', label='average FR of {} cell euclidean radius from the {}th cell'.format(radius, neuron))
            axes_list[subplot].plot(FR_mean, '+:k', label='{}th cell'.format(neuron))
            axes_list[subplot].legend()
            axes_list[subplot].set_title(titles_list[subplot], y=1.10, fontsize=14)
            axes_list[subplot].set_ylabel('[Hz]', fontsize=14)
            axes_list[subplot].set_xlabel('[ms]', fontsize=14)

        def plot_cellPosition(FR, neuron, subplot):
            from matplotlib.colors import ListedColormap
            cmapPos = ListedColormap(['w', 'y', 'r'])
            pop = self.args.N_pop
            FR_subset = []
            radius = 2
            for j in range(len(FR)):
                Ax = neuron // int(np.sqrt(pop))
                Ay = neuron % int(np.sqrt(pop))
                Bx = j // int(np.sqrt(pop))
                By = j % int(np.sqrt(pop))
                d = np.sqrt((Ax - Bx)**2 + (Ay - By)**2) #abs(i-j)
                if d <= radius:
                        FR_subset.append(j)

            mat = np.zeros((int(np.sqrt(pop)),int(np.sqrt(pop))))
            for subset in FR_subset:
                if subset == neuron:
                    Cx = subset//int(np.sqrt(pop))
                    Cy = subset%int(np.sqrt(pop))
                    mat[Cx,Cy] = mat[Cx,Cy] + 2
                else:
                    Cx = subset//int(np.sqrt(pop))
                    Cy = subset%int(np.sqrt(pop))
                    mat[Cx,Cy] = mat[Cx,Cy] + 1

            axes_list[subplot].matshow(mat, cmap=cmapPos)
            axes_list[subplot].set_title(titles_list[subplot] + '{}th (red) in 2D space \n and its {} cell radius (yellow)'.format(neuron, radius), y=1.10, fontsize=14)
            axes_list[subplot].set_ylabel('neurons', fontsize=14)
            axes_list[subplot].set_xlabel('neurons', fontsize=14)

        titles_list = ['Position of the',
                       'Position of the',
                       'Position of the',
                      'Input Short Range',
                      'Input Medium Range',
                      'Input Long Range',
                      'Exc Layer Range',
                      'Exc Medium Range',
                      'Exc Long Range',
                      'Inh Short Range',
                      'Inh Medium Range',
                      'Inh Long Range']


        for subplot in range(len(titles_list)):
            if subplot == 0:
                plot_cellPosition(fr[0], nS, subplot)
            if subplot == 1:
                plot_cellPosition(fr[0], nM, subplot)
            if subplot == 2:
                plot_cellPosition(fr[0], nL, subplot)
            if subplot == 3:
                 plot_meanFR(fr[0], nS,  subplot)
            if subplot == 4:
                 plot_meanFR(fr[0], nM,  subplot)
            if subplot == 5:
                 plot_meanFR(fr[0], nL,  subplot)
            if subplot == 6:
                 plot_meanFR(fr[1], nS,  subplot)
            if subplot == 7:
                 plot_meanFR(fr[1], nM,  subplot)
            if subplot == 8:
                 plot_meanFR(fr[1], nL,  subplot)
            if subplot == 9:
                 plot_meanFR(fr[2], nS,  subplot)
            if subplot == 10:
                 plot_meanFR(fr[2], nM,  subplot)
            if subplot == 11:
                 plot_meanFR(fr[2], nL,  subplot)

            fig.savefig(os.path.join(self.figpath, 'plot_figure2old.pdf'))


    def plot_abstract_voltages(self, fr, showArg=True):

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import cm
        import numpy as np
        cmap = cm.PRGn

        rows = 2
        cols = 4
        fig, axes = plt.subplots(rows,cols,figsize=(20,15))
        fig.tight_layout(pad=0.4)
        axes_list = fig.axes
        fig.suptitle('Voltage Window Snapshots')

        titles_list = [
                      'V 0:250 | E layer',
                      'V 250:500 E layer',
                      'V 500:750 | E layer',
                      'V 750:1000 | E layer',
                      'I fr 0:250 | I layer',
                      'I fr 250:500 | I layer',
                      'I fr 500:750 | I layer',
                      'I fr 750:1000 | I layer']

        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                            wspace=0.35)

        #dim = int(np.sqrt(len(fr[0])))
        for i in range(len(titles_list)):
            if i==0:
                im = axes_list[i].plot(self.E_v.T[0:250])
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=16)
                axes_list[i].set_ylabel('[mV]', fontsize=14)
                axes_list[i].set_xlabel('[ms]', fontsize=14)
                axes_list[i].set_xticks(np.arange(0, 251, step=50))
                axes_list[i].set_xticklabels(['0', '50', '100', '150', '200', '250'])


            if i==1:
                im = axes_list[i].plot(self.E_v.T[250:500])
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=16)
                axes_list[i].set_ylabel('[mV]', fontsize=14)
                axes_list[i].set_xlabel('[ms]', fontsize=14)
                axes_list[i].set_xticks(np.arange(0, 251, step=50))
                axes_list[i].set_xticklabels(['250', '300', '350', '400', '450', '500'])

            if i==2:
                im = axes_list[i].plot(self.E_v.T[500:750])
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=16)
                axes_list[i].set_ylabel('[mV]', fontsize=14)
                axes_list[i].set_xlabel('[ms]', fontsize=14)
                axes_list[i].set_xticks(np.arange(0, 251, step=50))
                axes_list[i].set_xticklabels(['500', '550', '600', '650', '700', '750'])


            if i==3:
                im = axes_list[i].plot(self.E_v.T[750:1000])
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=16)
                axes_list[i].set_ylabel('[mV]', fontsize=14)
                axes_list[i].set_xlabel('[ms]', fontsize=14)
                axes_list[i].set_xticks(np.arange(0, 251, step=50))
                axes_list[i].set_xticklabels(['750', '800', '850', '900', '950', '1000'])

            if i==4:
                im = axes_list[i].plot(self.I_v.T[0:250])
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=16)
                axes_list[i].set_ylabel('[mV]', fontsize=14)
                axes_list[i].set_xlabel('[ms]', fontsize=14)
                axes_list[i].set_xticks(np.arange(0, 251, step=50))
                axes_list[i].set_xticklabels(['0', '50', '100', '150', '200', '250'])

            if i==5:
                im = axes_list[i].plot(self.I_v.T[250:500])
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=16)
                axes_list[i].set_ylabel('[mV]', fontsize=14)
                axes_list[i].set_xlabel('[ms]', fontsize=14)
                axes_list[i].set_xticks(np.arange(0, 251, step=50))
                axes_list[i].set_xticklabels(['250', '300', '350', '400', '450', '500'])

            if i==6:
                im = axes_list[i].plot(self.I_v.T[500:750])
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=16)
                axes_list[i].set_ylabel('[mV]', fontsize=14)
                axes_list[i].set_xlabel('[ms]', fontsize=14)
                axes_list[i].set_xticks(np.arange(0, 251, step=50))
                axes_list[i].set_xticklabels(['500', '550', '600', '650', '700', '750'])

            if i==7:
                im = axes_list[i].plot(self.I_v.T[750:1000])
                axes_list[i].set_title(titles_list[i], y=1.10, fontsize=16)
                axes_list[i].set_ylabel('[mV]', fontsize=14)
                axes_list[i].set_xlabel('[ms]', fontsize=14)
                axes_list[i].set_xticks(np.arange(0, 251, step=50))
                axes_list[i].set_xticklabels(['750', '800', '850', '900', '950', '1000'])

        fig.savefig(os.path.join(self.figpath, 'plot_abstract_voltages.pdf'))


    def plot_figure2(self, nS, nM, nL, showArg=True):

        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np

         # TAKE THE FIRING RATE
        S_spikes = self.spikeSources.segments[0].spiketrains
        E_spikes = self.popEdata.segments[0].spiketrains
        I_spikes = self.popIdata.segments[0].spiketrains

        # FIRING RATE FUNCTION DEFINITION
        N_pop = self.args.N_pop
        time_bins = self.time_bins
        N_tau = self.args.N_tau
        dt = 1/ time_bins
        def firing_rate(b, N_tau=N_tau) :
            fr = np.zeros_like(b).astype(float)
            for t in range(b.shape[0]) :
            #print(1 - 1 / tau, t, fr.mean())
                fr[t, :] = (1 - 1 / N_tau) * fr[t-1, :] + 1 / N_tau * b[t, :]
            return fr.T # transposition to have (N,T)

        # FIRING RATES OF SOURCES SPIKES
        S_binMat = np.zeros((time_bins, N_pop), dtype='int')
        S_spk_list = []
        for i in range(N_pop): # loops over sources
            S_spike_train = S_spikes[i].as_array() # makes as array each source
            S_spike_times = [int(t) for t in S_spike_train]
            S_spk_list.append(S_spikes[i].as_array())
            for spike_time in S_spike_times: # loops over spikes for each source
                    #print(spike_time, i)
                    S_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                    # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        S_fr = firing_rate(S_binMat)*(1/dt)

        # FIRING RATES OF E LAYER SPIKES
        E_binMat = np.zeros((time_bins, N_pop), dtype='int')
        E_spk_list = []
        for i in range(N_pop): # loops over sources
            E_spike_train = E_spikes[i].as_array() # makes as array each source
            E_spk_list.append(E_spikes[i].as_array())
            E_spike_times = [int(t) for t in E_spike_train]
            for spike_time in E_spike_times: # loops over spikes for each source
                #print(spike_time, i)
                E_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        E_fr = firing_rate(E_binMat)*(1/dt)

        # FIRING RATES OF I LAYER SPIKES
        I_binMat = np.zeros((time_bins, N_pop), dtype='int')
        I_spk_list = []
        for i in range(N_pop): # loops over sources
            I_spike_train = I_spikes[i].as_array() # makes as array each source
            I_spk_list.append(I_spikes[i].as_array())
            I_spike_times = [int(t) for t in I_spike_train]
            for spike_time in I_spike_times: # loops over spikes for each source
                #print(spike_time, i)
                I_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        I_fr = firing_rate(I_binMat)*(1/dt)

        # FIRING RATES OF SOURCE, EXC AND INH LAYER
        fr = [S_fr, E_fr, I_fr]

        # MAKE THE PLOTS
        cmap = cm.PRGn
        rows = 1
        cols = 3
        fig, axes = plt.subplots(rows,cols,figsize=(13,5), sharey=True)
        fig.tight_layout(pad=0.4)
        axes_list = fig.axes
        #fig.suptitle('Averaged firing rate of neurons in relation to specific input space location',  fontsize=16)
        fig.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.95, hspace=0.25,
                            wspace=0.35)

        def plot_meanFR(fr, neuron, subplot):

            pop = self.args.N_pop
            FR_input = fr[0]
            FR_exc = fr[1]
            FR_inh = fr[2]

            FR_subset = []
            radius = 2
            for j in range(self.time_bins):
                Ax = neuron // int(np.sqrt(pop))
                Ay = neuron % int(np.sqrt(pop))
                Bx = j // int(np.sqrt(pop))
                By = j % int(np.sqrt(pop))
                d = np.sqrt((Ax - Bx)**2 + (Ay - By)**2) #abs(i-j)
                if d <= radius:
                        FR_subset.append(j)

            FR_input_mean = []
            FR_exc_mean = []
            FR_inh_mean = []
            instants = [350, 500, 750]


            for fr in range(self.time_bins):
                    FR_input_mean.append(np.mean(FR_input[FR_subset].T[fr]))
                    FR_exc_mean.append(np.mean(FR_exc[FR_subset].T[fr]))
                    FR_inh_mean.append(np.mean(FR_inh[FR_subset].T[fr]))

            if subplot == 0:
                axes_list[subplot].plot(FR_input_mean, '-k', label='Input layer') # | {}th cell'.format(neuron))
                axes_list[subplot].plot(FR_exc_mean, '-r', label=' Exc layer') #| {}th cell'.format(neuron))
                axes_list[subplot].plot(FR_inh_mean, '-b', label='Inh layer') # | {}th cell'.format(neuron))
                axes_list[subplot].axvline(x=instants[subplot])
                axes_list[subplot].legend()
            else:
                axes_list[subplot].plot(FR_input_mean, '-k')#, label='Input layer') # | {}th cell'.format(neuron))
                axes_list[subplot].plot(FR_exc_mean, '-r')#, label=' Exc layer') #| {}th cell'.format(neuron))
                axes_list[subplot].plot(FR_inh_mean, '-b')#, label='Inh layer') # | {}th cell'.format(neuron))
                axes_list[subplot].axvline(x=instants[subplot])


            axes_list[subplot].set_title(titles_list[subplot], y=1.10, fontsize=14)
            axes_list[subplot].set_ylabel('Firing Rate [Hz]', fontsize=14)
            axes_list[subplot].set_xlabel('Time [ms]', fontsize=14)

        def plot_cellPosition(FR, neuron, subplot):
            from matplotlib.colors import ListedColormap
            cmapPos = ListedColormap(['w', 'y', 'r'])
            pop = self.args.N_pop
            FR_subset = []
            radius = 2
            for j in range(self.time_bins):
                Ax = neuron // int(np.sqrt(pop))
                Ay = neuron % int(np.sqrt(pop))
                Bx = j // int(np.sqrt(pop))
                By = j % int(np.sqrt(pop))
                d = np.sqrt((Ax - Bx)**2 + (Ay - By)**2) #abs(i-j)
                if d <= radius:
                        FR_subset.append(j)

            mat = np.zeros((int(np.sqrt(pop)),int(np.sqrt(pop))))
            for subset in FR_subset:
                if subset == neuron:
                    Cx = subset//int(np.sqrt(pop))
                    Cy = subset%int(np.sqrt(pop))
                    mat[Cx,Cy] = mat[Cx,Cy] + 2
                else:
                    Cx = subset//int(np.sqrt(pop))
                    Cy = subset%int(np.sqrt(pop))
                    mat[Cx,Cy] = mat[Cx,Cy] + 1

            axes_list[subplot].matshow(mat, cmap=cmapPos)
            axes_list[subplot].set_title(titles_list[subplot] + '{}th (red) in 2D space \n and its {} cell radius (yellow)'.format(neuron, radius), y=1.10, fontsize=14)
            axes_list[subplot].set_ylabel('neurons', fontsize=14)
            axes_list[subplot].set_xlabel('neurons', fontsize=14)

        titles_list = ['Short Trajectory', 'Medium Trajectory', 'Long Trajectory']


        for subplot in range(len(titles_list)):
            if subplot == 0:
                plot_meanFR(fr, nS, subplot)
            if subplot == 1:
                plot_meanFR(fr, nM, subplot)
            if subplot == 2:
                plot_meanFR(fr, nL, subplot)


        fig.savefig(os.path.join(self.figpath, 'plot_figure2.pdf'))


    def plot_figure3(self, perStart, perEnd, showArg=True):

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import cm
        import numpy as np

        # TAKE THE FIRING RATE
        S_spikes = self.spikeSources.segments[0].spiketrains
        E_spikes = self.popEdata.segments[0].spiketrains
        I_spikes = self.popIdata.segments[0].spiketrains

        # FIRING RATE FUNCTION DEFINITION
        N_pop = self.args.N_pop
        time_bins = self.time_bins
        N_tau = self.args.N_tau
        dt = 1/ time_bins
        def firing_rate(b, N_tau=N_tau) :
            fr = np.zeros_like(b).astype(float)
            for t in range(b.shape[0]) :
            #print(1 - 1 / tau, t, fr.mean())
                fr[t, :] = (1 - 1 / N_tau) * fr[t-1, :] + 1 / N_tau * b[t, :]
            return fr.T # transposition to have (N,T)

        # FIRING RATES OF SOURCES SPIKES
        S_binMat = np.zeros((time_bins, N_pop), dtype='int')
        S_spk_list = []
        for i in range(N_pop): # loops over sources
            S_spike_train = S_spikes[i].as_array() # makes as array each source
            S_spike_times = [int(t) for t in S_spike_train]
            S_spk_list.append(S_spikes[i].as_array())
            for spike_time in S_spike_times: # loops over spikes for each source
                    #print(spike_time, i)
                    S_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                    # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        S_fr = firing_rate(S_binMat)*(1/dt)

        # FIRING RATES OF E LAYER SPIKES
        E_binMat = np.zeros((time_bins, N_pop), dtype='int')
        E_spk_list = []
        for i in range(N_pop): # loops over sources
            E_spike_train = E_spikes[i].as_array() # makes as array each source
            E_spk_list.append(E_spikes[i].as_array())
            E_spike_times = [int(t) for t in E_spike_train]
            for spike_time in E_spike_times: # loops over spikes for each source
                #print(spike_time, i)
                E_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        E_fr = firing_rate(E_binMat)*(1/dt)

        # FIRING RATES OF I LAYER SPIKES
        I_binMat = np.zeros((time_bins, N_pop), dtype='int')
        I_spk_list = []
        for i in range(N_pop): # loops over sources
            I_spike_train = I_spikes[i].as_array() # makes as array each source
            I_spk_list.append(I_spikes[i].as_array())
            I_spike_times = [int(t) for t in I_spike_train]
            for spike_time in I_spike_times: # loops over spikes for each source
                #print(spike_time, i)
                I_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
        # translating firing rate output in Hz
        I_fr = firing_rate(I_binMat)*(1/dt)

        # FIRING RATES OF SOURCE, EXC AND INH LAYER
        fr = [S_fr, E_fr, I_fr]


        ###---LOADING INPUT -------------------------------------------------------------------------------
        self.time_bins = int(self.args.simtime/self.args.dt) #extended time
        #extended pixels
        width = int(np.sqrt(self.args.N_pop))
        #1.1 generate 2d with time stimulus
        inputBar = generate_dot(width, width, self.time_bins,
                                dot_size=0.1,
                                flash_start=0.2,
                                flash_duration=0.6,
                                im_noise=self.args.im_noise,
                                im_contrast=self.args.im_contrast,
                                )
        inputBar = inputBar.transpose(2, 1, 0)

        def plot_figure3_fr(fig, axes_list, figIndex, colorMap, titles_list, i, periods, fr):
            dim = int(np.sqrt(self.args.N_pop))
            In_maxFr = []
            for j in range(self.args.N_pop):
                In_maxFr.append(fr[j, int(periods[0]):int(periods[1])].mean())

            im = axes_list[figIndex].imshow(np.asarray(In_maxFr).reshape(dim,dim), cmap=colorMap, vmin=0, vmax=200)
            axes_list[figIndex].set_title(titles_list[i], y=1.10, fontsize=14)
            #axes_list[figIndex].set_ylabel('', fontsize=14)
            axes_list[figIndex].set_xticks([])
            axes_list[figIndex].set_yticks([])
            cbar = fig.colorbar(im, ax=axes_list[figIndex],fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('[Hz]')


        def plot_figure3_gen(fig, titles_list, perStart, perEnd, figIndex, fr):
                 # MAKE THE PLOTS

            periods = [perStart, perEnd]
            from scipy import stats as st
            if (figIndex == 0):
                instant = int(np.mean(periods))
                #colorMap = 'brg'
                im = axes_list[figIndex].imshow(st.zscore(inputBar[instant,:,:]), vmin=-3, vmax=3)#, cmap=colorMap)
                axes_list[figIndex].set_title(titles_list[figIndex], y=1.10, fontsize=14)
                axes_list[figIndex].set_ylabel('stimulus', fontsize=14)
                axes_list[figIndex].set_xlabel('', fontsize=14)
                axes_list[figIndex].set_xticks([])
                axes_list[figIndex].set_yticks([])
                cbar = fig.colorbar(im, ax=axes_list[figIndex],fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel('[Z-lumen]')

            if (figIndex == 1):
                i = figIndex
                fr = fr[0]
                colorMap = 'binary'
                axes_list[figIndex].set_ylabel('LGN layer', fontsize=14)
                plot_figure3_fr(fig, axes_list, figIndex, colorMap, titles_list, i, periods, fr)

            if (figIndex == 2):
                i = figIndex
                fr = fr[1]
                colorMap = 'Reds'
                axes_list[figIndex].set_ylabel('excitatory layer', fontsize=14)
                plot_figure3_fr(fig, axes_list, figIndex, colorMap, titles_list, i, periods, fr)

            if (figIndex == 3):
                i = figIndex
                fr = fr[2]
                colorMap = 'Blues'
                axes_list[figIndex].set_ylabel('inhibitory layer', fontsize=14)
                plot_figure3_fr(fig, axes_list, figIndex, colorMap, titles_list, i, periods, fr)

        # generates plot
        cmap = cm.PRGn
        rows = 4
        cols = 1
        fig, axes = plt.subplots(rows,cols,figsize=(6.5,8.0))
        axes_list = fig.axes
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)#(pad=0.1)#(pad=0.4)
        fig.subplots_adjust(
                top=0.92,
                bottom=0.08,
                left=0.10, #0.10,
                right=0.35, #0.95,
                hspace=0.05,
                wspace=0.15) #0.35

        if showArg == 'False':
            plt.close(fig)
        if showArg == 'True':
            plt.show()

        titles_list = [
                      '{}:{} [ms]'.format(perStart, perEnd),
                      '', '', '']

        for tit in range(len(titles_list)):
            plot_figure3_gen(fig, titles_list, perStart, perEnd, tit, fr)

        fig.savefig(os.path.join(self.figpath, 'plot_figure3_{:04d}-{:04d}.png'.format(perStart, perEnd)))




    def plot_videoSimulation(startVideo, endVideo, stepVideo):

        def plot_figure3(self, perStart, perEnd, showArg=True):

            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            from matplotlib import cm
            import numpy as np

            # TAKE THE FIRING RATE
            S_spikes = self.spikeSources.segments[0].spiketrains
            E_spikes = self.popEdata.segments[0].spiketrains
            I_spikes = self.popIdata.segments[0].spiketrains

            # FIRING RATE FUNCTION DEFINITION
            N_pop = self.args.N_pop
            time_bins = self.time_bins
            N_tau = self.args.N_tau
            dt = 1/ time_bins
            def firing_rate(b, N_tau=N_tau) :
                fr = np.zeros_like(b).astype(float)
                for t in range(b.shape[0]) :
                #print(1 - 1 / tau, t, fr.mean())
                    fr[t, :] = (1 - 1 / N_tau) * fr[t-1, :] + 1 / N_tau * b[t, :]
                return fr.T # transposition to have (N,T)

            # FIRING RATES OF SOURCES SPIKES
            S_binMat = np.zeros((time_bins, N_pop), dtype='int')
            S_spk_list = []
            for i in range(N_pop): # loops over sources
                S_spike_train = S_spikes[i].as_array() # makes as array each source
                S_spike_times = [int(t) for t in S_spike_train]
                S_spk_list.append(S_spikes[i].as_array())
                for spike_time in S_spike_times: # loops over spikes for each source
                        #print(spike_time, i)
                        S_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                        # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
            # translating firing rate output in Hz
            S_fr = firing_rate(S_binMat)*(1/dt)

            # FIRING RATES OF E LAYER SPIKES
            E_binMat = np.zeros((time_bins, N_pop), dtype='int')
            E_spk_list = []
            for i in range(N_pop): # loops over sources
                E_spike_train = E_spikes[i].as_array() # makes as array each source
                E_spk_list.append(E_spikes[i].as_array())
                E_spike_times = [int(t) for t in E_spike_train]
                for spike_time in E_spike_times: # loops over spikes for each source
                    #print(spike_time, i)
                    E_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                    # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
            # translating firing rate output in Hz
            E_fr = firing_rate(E_binMat)*(1/dt)

            # FIRING RATES OF I LAYER SPIKES
            I_binMat = np.zeros((time_bins, N_pop), dtype='int')
            I_spk_list = []
            for i in range(N_pop): # loops over sources
                I_spike_train = I_spikes[i].as_array() # makes as array each source
                I_spk_list.append(I_spikes[i].as_array())
                I_spike_times = [int(t) for t in I_spike_train]
                for spike_time in I_spike_times: # loops over spikes for each source
                    #print(spike_time, i)
                    I_binMat[spike_time-1, i-1] = 1 # the -1 in the args traslates the values in list indeces
                                                    # (e.g. 50th cell spikes at 1000ms -> E_binMat[49, 999])
            # translating firing rate output in Hz
            I_fr = firing_rate(I_binMat)*(1/dt)

            # FIRING RATES OF SOURCE, EXC AND INH LAYER
            fr = [S_fr, E_fr, I_fr]


            ###---LOADING INPUT -------------------------------------------------------------------------------
            self.time_bins = int(self.args.simtime/self.args.dt) #extended time
            #extended pixels
            width = int(np.sqrt(self.args.N_pop))
            #1.1 generate 2d with time stimulus
            inputBar = generate_dot(width, width, self.time_bins,
                                    dot_size=0.1,
                                    flash_start=0.2,
                                    flash_duration=0.6,
                                    im_noise=self.args.im_noise,
                                    im_contrast=self.args.im_contrast,
                                    )
            inputBar = inputBar.transpose(2, 1, 0)

            def plot_figure3_fr(fig, axes_list, figIndex, colorMap, titles_list, i, periods, fr):
                dim = int(np.sqrt(self.args.N_pop))
                In_maxFr = []
                for j in range(self.args.N_pop):
                    In_maxFr.append(fr[j, int(periods[0]):int(periods[1])].mean())

                im = axes_list[figIndex].imshow(np.asarray(In_maxFr).reshape(dim,dim), cmap=colorMap, vmin=0, vmax=200)
                axes_list[figIndex].set_title(titles_list[i], y=1.10, fontsize=14)
                #axes_list[figIndex].set_ylabel('', fontsize=14)
                axes_list[figIndex].set_xticks([])
                axes_list[figIndex].set_yticks([])
                cbar = fig.colorbar(im, ax=axes_list[figIndex],fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel('[Hz]')


            def plot_figure3_gen(fig, titles_list, perStart, perEnd, figIndex, fr):
                     # MAKE THE PLOTS

                periods = [perStart, perEnd]

                if (figIndex == 0):
                    instant = int(np.mean(periods))
                    colorMap = 'plasma'
                    im = axes_list[figIndex].imshow(inputBar[instant,:,:], cmap=colorMap, vmin=0, vmax=0.5)
                    axes_list[figIndex].set_title(titles_list[figIndex], y=1.10, fontsize=14)
                    axes_list[figIndex].set_ylabel('stimulus', fontsize=14)
                    axes_list[figIndex].set_xlabel('', fontsize=14)
                    axes_list[figIndex].set_xticks([])
                    axes_list[figIndex].set_yticks([])
                    cbar = fig.colorbar(im, ax=axes_list[figIndex],fraction=0.046, pad=0.04)
                    cbar.ax.set_ylabel('[?]')

                if (figIndex == 1):
                    i = figIndex
                    fr = fr[0]
                    colorMap = 'binary'
                    axes_list[figIndex].set_ylabel('input layer', fontsize=14)
                    plot_figure3_fr(fig, axes_list, figIndex, colorMap, titles_list, i, periods, fr)

                if (figIndex == 2):
                    i = figIndex
                    fr = fr[1]
                    colorMap = 'Reds'
                    axes_list[figIndex].set_ylabel('excitatory layer', fontsize=14)
                    plot_figure3_fr(fig, axes_list, figIndex, colorMap, titles_list, i, periods, fr)

                if (figIndex == 3):
                    i = figIndex
                    fr = fr[2]
                    colorMap = 'Blues'
                    axes_list[figIndex].set_ylabel('inhibitory layer', fontsize=14)
                    plot_figure3_fr(fig, axes_list, figIndex, colorMap, titles_list, i, periods, fr)

            # generates plot
            cmap = cm.PRGn
            rows = 4
            cols = 1
            fig, axes = plt.subplots(rows,cols,figsize=(5,8))
            axes_list = fig.axes
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)#(pad=0.1)#(pad=0.4)
            fig.subplots_adjust(top=0.92, bottom=0.08,
                    left=0.10, #0.10,
                    right=0.95, #0.95,
                    hspace=0.05,
                    wspace=0.5) #0.35

            if showArg == 'False':
                plt.close(fig)
            if showArg == 'True':
                plt.show()

            titles_list = [
                          '{}:{} [ms]'.format(perStart, perEnd),
                          '', '', '']

            for tit in range(len(titles_list)):
                plot_figure3_gen(fig, titles_list, perStart, perEnd, tit, fr)

            fig.savefig(os.path.join(self.figpath, 'plot_figure3_{:04d}-{:04d}.png'.format(perStart, perEnd)))




        import numpy as np
        simPeriods = np.arange(startVideo, endVideo, stepVideo)
        for startingTime in simPeriods:
                endingTime = startingTime + stepVideo
                plot_figure3(startingTime, endingTime,'False')

        import subprocess
        import os
        path = os.path.join(self.figpath)
        i = path + '/*.png'
        ogif = path + '/animation_{:04d}-{:04d}-{:04}.gif'.format(startVideo, endVideo, stepVideo)
        omp4 = path + '/video_{:04d}-{:04d}-{:04}.mp4'.format(perStart, perEnd, stepVideo)
        subprocess.call('convert -delay 125 -loop 0 ' + i + ' ' + ogif, shell = True)
        #subprocess.call('ffmpeg -i ' + ogif  ' -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"' + ' ' + omp4, shell = True)
        #ffmpeg -i animation.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" video.mp4
        # convert -delay 125 -loop 0 *figure3* animation.gif
        subprocess.call('rm ' + path + '*plot_figure3_*.png', shell = True)

