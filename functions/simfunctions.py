import sund
import numpy as np
from contextlib import contextmanager
import os
import copy
import json
from json import JSONEncoder
from scipy.signal import find_peaks

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

ROUNDING_PRECISION = 15

def calculate_norm_factor(T,k_syst,k_diast,m1,m2):
    # to make sure that Emax is maximum of elastance curve and emin is minimum of elastance curve
    N=200 #Length of the time axis vector
    T_axis= np.linspace(0,T,N)
    g1 = np.divide(T_axis, k_syst*T)**m1
    g2 = np.power(np.divide(T_axis , k_diast*T), m2)
    prod = (g1 / (1 + g1)) * (1 / (1 + g2))
    norm_factor = max(prod)

    return norm_factor


########################################################################################################################
######################### SIMULATION SETUP FUNCTIONS ###################################################################
########################################################################################################################

def create_shared_time_vector(data, time_unit='s'):
    """
    Create a unified time vector that includes all time points needed for all subjects.
    
    Args:
        data: Dictionary of subject data containing 'simulationtime' for each subject
        time_unit: Time unit for the simulation ('s' or 'm')
    
    Returns:
        numpy array: Unified time vector containing all unique time points from all subjects
    """
    all_times = []
    
    for subj in data:
        if 'input' in data[subj] and 'simulationtime' in data[subj]['input']:
            times = data[subj]['input']['simulationtime']
            if isinstance(times, (list, np.ndarray)):
                all_times.extend(times)
            else:
                all_times.append(times)
    
    if not all_times:
        # Return default time vector if no simulation times found
        if time_unit == 's':
            return np.arange(0, 1, 0.0001)
        else:  # minutes
            return np.arange(0, 10, 0.01)
    
    # Remove duplicates and sort
    unique_times = np.unique(all_times)
    return unique_times

def add_exercise_inputs(datainput,ex,addexercise=True):
    from functions.utils import get_BSA
    # ex: exercise activity object
    # datainput: input dict based on data containing information for the simulation

    if addexercise:
        # Set exercise input. percWmax is prioritized, but if neither percWmax or wmax is given in data, pure watt is used and Wmax should be estimated as a parameter.
        if 'percWmax' in  datainput.keys() and datainput['percWmax'] is not None:
            percWmax = np.append(0,datainput['percWmax'])
            ex.add_output(type="piecewise_constant", name="percWmax_in", t = datainput['time'], f=percWmax)
            ex.add_output(type="constant", name="watt_in", f=0)
        elif 'watt' in datainput.keys() and 'wmax' in  datainput.keys()  and datainput['wmax'] is not None:
            watt = datainput['watt']
            wmax = datainput['wmax']
            percWmax = np.divide(watt,wmax)*100
            percWmax = np.insert(percWmax,0,0)
            ex.add_output(type="piecewise_constant", name="percWmax_in", t = datainput['time'], f=percWmax)
            ex.add_output(type="constant", name="watt_in", f=0)
        elif 'watt' in datainput.keys():
            watt = [0]+datainput['watt']
            ex.add_output(type="piecewise_constant", name="watt_in", t = datainput['time'], f=watt)
            ex.add_output(type="constant", name="percWmax_in", f=0)
        else:
            print('Warning: no watt or percWmax found in data input, creating simulation without exercise input')

    if 'sex' in datainput.keys():
        ex.add_output(type="constant", name="sex",f=datainput['sex'])
    else:
        ex.add_output(type="constant", name="sex",f=1) #1=male 0 =female
    if 'age' in datainput.keys():
        ex.add_output(type="constant", name="age",f=datainput['age'])
    else:
        ex.add_output(type="constant", name="age",f=60)

    if 'Hb' in datainput.keys():
        ex.add_output(type="constant", name="hb",f=datainput['Hb'])
    else:
        ex.add_output(type="constant", name="hb",f=150) # g/L

    if 'HRmax' in datainput.keys() and datainput['HRmax'] is not None:
        ex.add_output(type="constant", name="HRmax_in",f=datainput['HRmax'])
    elif 'HRmax_upright'  in datainput and datainput['HRmax_upright'] is not None:
        #HRmax in supine position is estimated to be 0.85 percent lower than HRmax upright
        ex.add_output(type="constant", name="HRmax_in",f=datainput['HRmax_upright']*0.85)
    elif 'age'  in datainput:
        ex.add_output(type="constant", name="HRmax_in",f=(220-datainput['age'])*0.85) 
    else:
        ex.add_output(type="constant", name="HRmax_in",f=1) #set inside the model to (220-age) * 0.85
    
    if 'weight' in datainput.keys() and isinstance(datainput['weight'],float):
        ex.add_output(type="constant", name="weight",f=datainput['weight'])
        w=datainput['weight']
    else:
        ex.add_output(type="constant", name="weight", f=70)
        w=70
    if 'height' in datainput.keys() and isinstance(datainput['height'],float):
        ex.add_output(type="constant", name="height",f=datainput['height'])
        h=datainput['height']
    else:
        ex.add_output(type="constant", name="height", f=1.75)
        h=1.75
    BSA = get_BSA(datainput)
    ex.add_output(type="constant", name="BSA",f=BSA)

def setup_exercise_simulation(modelname,data=None,parametervalues=None):
    # Set up the model
    try:
        sund.install_model("models/" + modelname + ".txt")
    except Exception as e:
        print(f"Error installing model {modelname}: {e}")
        #raise
    M = sund.import_model(modelname)
    exercisemodel = M()

    # Create simulation objects with activities based on the data
    if data is None:
        sims = sund.Simulation(models = exercisemodel, time_unit='m',time_vector = np.arange(0,10,0.01))
        simsSS  = sund.Simulation(models = exercisemodel, time_unit='m',time_vector = [0,10])
    else:
        sims = {}
        simsSS = {}
        # set inputs accodring to data if it exists, otherwise to default value
        for idx, experiment in enumerate(data):
            ex = sund.Activity(time_unit='m')
            add_exercise_inputs(data[experiment]['input'],ex)
            sims[experiment] = sund.Simulation(models = exercisemodel, activities = ex,
                                                time_unit='m',time_vector=data[experiment]['input']['simulationtime'],augment_time_vector=False)
            rest = sund.Activity(time_unit='m')
            add_exercise_inputs(data[experiment]['input'],rest,addexercise=False)
            simsSS[experiment] = sund.Simulation(models = exercisemodel, activities = rest,
                                                time_unit='m',time_vector=[-1000,-60],augment_time_vector=False)
    # Parameter vector for each experiment/subject
    if data is None:
        if parametervalues is None:
            params = exercisemodel.parameter_values
        else:
            params=parametervalues
    else:
        params = {}
        for exp in data:
            if parametervalues is None:
                params[exp] = copy.deepcopy(exercisemodel.parameter_values)
            else:
                params[exp] = parametervalues
            # Add parameters based on data (initial conditions and wmax etc)
            set_IC_params(params[exp],data[exp],sims[exp].parameter_names)#set initial condition parameters according to data

    return exercisemodel, sims, params, simsSS

def setup_avatarsimulation_data(modelname,data=None,load_params=False,activity='Rest'):
    # Set up the model
    sund.install_model("models/" + modelname + ".txt")
    M = sund.import_model(modelname)
    avatarmodel = M() # create a model object

    # set default parameter values
    Cpvc = 1
    Rpu = 0.01000000000000000021
    Rpv = 0.00200000000000000004
    Lpv = 0.00050000000000000001
    Rtot = 0.94663323540623101060
    Ctot = 1.46726648036775930706
    ELCo = 3.81156246548926258555
    Caa = 0.12367084767074999763
    Emax_LA = 0.14007262281514298863
    Emax_LV = 1.92293715777547391532
    Emin_LA = 0.11564247837264500507
    Emin_LV = 0.07564372002459850142
    Lao = 0.00040221336467217002
    Lav = 0.00014603903586280601
    Lmv = 0.00050166636372005701
    Ppu = 9.64685207296152036349
    Rao = 0.07864610756126340363
    Rmv = 0.00434744560939943989
    k_diast_LA = 0.19828852451323000783
    k_diast_LV = 0.46001328156847398620
    k_syst_LA = 0.12359464349113100101
    k_syst_LV = 0.49334575719769002689
    m1_LA = 1.41118010206129995687
    m1_LV = 1.22911354922997007044
    m2_LA = 13.6
    m2_LV = 37.8
    onset_LA = 0.2325
    onset_LV = 1.4551
    V0_LA = 3.0
    V0_LV = 10.0
    tdiast = 0.3073
    Ks_LA = 0.00000001
    Ks_LV = 0.000000004
    RLAvisc = 0.0001
    RLVvisc = 0.0001
    Raa = 0.01
    Rpc = 0.01
    Rpvc = 0.01
    rho_blood = 1.06

    # calculate norm factors
    T=0.86
    norm_factor_LA=calculate_norm_factor(T,k_syst_LA,k_diast_LA,m1_LA,m2_LA)
    norm_factor_LV=calculate_norm_factor(T,k_syst_LV,k_diast_LV,m1_LA,m2_LV)

    onset_LA = 0.2326
    onset_LV = 1.455-1.5

    paramsavatar = [Cpvc, Rpu, Rpv, Lpv, Rtot, Ctot, ELCo, Caa, Emax_LA, Emax_LV, Emin_LA, Emin_LV, Lao, Lav, Lmv, Ppu, Rao, Rmv, k_diast_LA, k_diast_LV, k_syst_LA, k_syst_LV, m1_LA, m1_LV, m2_LA, m2_LV, onset_LA, onset_LV, V0_LA, V0_LV, tdiast, Ks_LA, Ks_LV, RLAvisc, RLVvisc, Raa, Rpc, Rpvc, rho_blood, norm_factor_LA, norm_factor_LV]

    khr_kdiastLV = 0.2
    khr_EmaxLV = 1
    khr_onsetLA = 0.1
    khr_R = 0.2

    currT = T
    t0 = 0
    Torig = T
    paramsavatar = np.append(paramsavatar,[khr_kdiastLV,khr_EmaxLV,khr_onsetLA,currT,t0,Torig])
    paramnamesavatar = ['Cpvc', 'Rpu', 'Rpv', 'Lpv', 'Rtot', 'Ctot', 'ELCo', 'Caa', 'Emax_LA', 'Emax_LV_orig', 
                        'Emin_LA', 'Emin_LV', 'Lao', 'Lav', 'Lmv','Ppu', 'Rao', 'Rmv','k_diast_LA', 'k_diast_LV_orig', 
                        'k_syst_LA', 'k_syst_LV', 'm1_LA', 'm1_LV', 'm2_LA', 'm2_LV', 'onset_LA_orig', 'onset_LV', 
                        'V0_LA', 'V0_LV', 'tdiast', 'Ks_LA', 'Ks_LV','RLAvisc', 'RLVvisc', 'Raa', 'Rpc', 'Rpvc',
                        'rho_blood', 'norm_factor_LA', 'norm_factor_LV','khr_kdiastLV','khr_EmaxLV','khr_onsetLA','currT','t0','Torig']

    parameter_values = avatarmodel.parameter_values
    for i,p in enumerate(avatarmodel.parameter_names):
        if p in paramnamesavatar:
            parameter_values[i] = paramsavatar[paramnamesavatar.index(p)]

    params = {}
    if data is None:
        params = parameter_values
    else:
        for subj in data:
            loadfromdata = True
            if loadfromdata:
                params[subj] = copy.deepcopy(parameter_values)
                if data[subj]['Emax_LV']['mean'] is not None:
                    if 'Emax_LV_orig' in avatarmodel.parameter_names:
                        params[subj][avatarmodel.parameter_names.index('Emax_LV_orig')] = data[subj]['Emax_LV']['mean'][0]
                    elif 'Emax_LV' in avatarmodel.parameter_names:
                        params[subj][avatarmodel.parameter_names.index('Emax_LV')] = data[subj]['Emax_LV']['mean'][0]
                if data[subj]['Caa']['mean'] is not None:
                    params[subj][avatarmodel.parameter_names.index('Caa')] = data[subj]['Caa']['mean'][0]
                if 'Ctot' in avatarmodel.parameter_names and 'Ctot' in data[subj] and data[subj]['Ctot']['mean'] is not None:
                    params[subj][avatarmodel.parameter_names.index('Ctot')] = data[subj]['Ctot']['mean'][0]
                if 'Rtot' in avatarmodel.parameter_names and 'Rtot' in data[subj] and data[subj]['Rtot']['mean'] is not None:
                    params[subj][avatarmodel.parameter_names.index('Rtot')] = data[subj]['Rtot']['mean'][0]
                elif 'currRtot' in avatarmodel.parameter_names and 'Rtot' in data[subj] and data[subj]['Rtot']['mean'] is not None:
                    params[subj][avatarmodel.parameter_names.index('currRtot')] = data[subj]['Rtot']['mean'][0]
                if data[subj]['ELCo']['mean'] is not None:
                    params[subj][avatarmodel.parameter_names.index('ELCo')] = data[subj]['ELCo']['mean'][0]
                if 'IC_Vlv' in avatarmodel.parameter_names and 'EDV' in data[subj]['input'] and data[subj]['input']['EDV']['mean'] is not None:
                    params[subj][avatarmodel.parameter_names.index('IC_Vlv')] = data[subj]['input']['EDV']['mean']
                else:
                    params[subj][avatarmodel.parameter_names.index('IC_Vlv')] =120 #mL
                if 'IC_Vaorta' in avatarmodel.parameter_names and 'Vaorta' in data[subj]['input'] and data[subj]['input']['Vaorta']['mean'] is not None:
                    params[subj][avatarmodel.parameter_names.index('IC_Vaorta')] = data[subj]['input']['Vaorta']['mean']
                else:
                    params[subj][avatarmodel.parameter_names.index('IC_Vaorta')] = 30 #mL of ascending aorta

                # set blood volume Vtot = claculated blood volume fr sex and weight/BMI/BSA, IC_Vlv = EDV, IC_Vla = min(Vla)
                if 'Vtot' in avatarmodel.parameter_names and 'Vaorta' in data[subj]['input'] and data[subj]['input']['Vaorta']['mean'] is not None:
                    if 'sex' in data[subj]['input'] and data[subj]['input']['sex'] is not None:
                        sex = data[subj]['input']['sex']
                    else:
                        sex=1 #1=male, 0=female
                    if 'weight' in data[subj]['input'] and data[subj]['input']['weight'] is not None:
                        weight = data[subj]['input']['weight']
                    else:
                        weight = 80
                    if 'height' in data[subj]['input'] and data[subj]['input']['height'] is not None:
                        height = data[subj]['input']['height']
                    else:
                        height = 1.75
                    #man = 1
                    Vtot = ((1-sex)*(0.3561 * height**3 + 0.03308 * weight + 0.1833) + sex*(0.3669 * height**3 + 0.03219 * weight + 0.6041))*1000 #Nadler's Equation for total blood volume 
                    params[subj][avatarmodel.parameter_names.index('Vtot')] = Vtot
                else:
                    params[subj][avatarmodel.parameter_names.index('Vtot')] = 5000

                if 'Vtot' in avatarmodel.parameter_names:
                    Vtot = params[subj][avatarmodel.parameter_names.index('Vtot')]
                    Vaorta = params[subj][avatarmodel.parameter_names.index('IC_Vaorta')]
                    Vrest = Vtot - Vaorta- params[subj][avatarmodel.parameter_names.index('IC_Vla')] -params[subj][avatarmodel.parameter_names.index('IC_Vlv')]
                    # Distribution of the rest of the total blood volume.The fractions should sum up to 1: 0.09+0.09+0.4+0.4+0.02 = 1
                    params[subj][avatarmodel.parameter_names.index('IC_Vpulm')] = 0.12*Vrest
                    params[subj][avatarmodel.parameter_names.index('IC_Vuart')] = 0.09*Vrest
                    params[subj][avatarmodel.parameter_names.index('IC_Vlart')] = 0.05*Vrest
                    params[subj][avatarmodel.parameter_names.index('IC_Vlvein')] = 0.27*Vrest
                    params[subj][avatarmodel.parameter_names.index('IC_Vuvein')] = 0.47*Vrest 

                if data[subj]['input']['T']['mean'] is not None:
                    Tsubj = data[subj]['input']['T']['mean']
                else:
                    Tsubj=copy.deepcopy(T)
                    print(f'ERROR in avatar setup: no T in data for subj {subj}. Using default T = {Tsubj}.')
                params[subj][avatarmodel.parameter_names.index('Torig')] = Tsubj
                params[subj][avatarmodel.parameter_names.index('currT')] = Tsubj
                # Calculate normalization factors for la and lv elastance and set onset LA based on onset LV
                set_avatarparams(params[subj],avatarmodel.parameter_names,onsetLV_sign=None)
        
    # Create simulation objects
    if data is None:
        #  Create activity to set T
        avataractivity = sund.Activity(time_unit='s')
        avataractivity.add_output("constant", "T_in", f=T)
        avataractivity.add_output("constant", "Rtot_in", f=1)
        sims = sund.Simulation(models = avatarmodel, activities=avataractivity,time_unit = 's',time_vector = np.arange(0,T+0.0001,0.0001)) 
        simsSS = sund.Simulation(models = avatarmodel, activities=avataractivity,time_unit = 's',time_vector = np.arange(0,T*25+0.0001,0.0001)) 
        # Set simulation settings
        sims.set_options({'abs_tol': 1e-16}) 
        sims.set_options({'rel_tol':1e-08})
        sims.set_options({'max_hnil':0}) 
        simsSS.set_options({'abs_tol': 1e-16}) 
        simsSS.set_options({'rel_tol':1e-08})
        simsSS.set_options({'max_hnil':0}) 
    else:
        sims = {}
        simsSS = {}
        for subj in data:
            # Create activity to set T
            avataractivity = sund.Activity(time_unit='s')
            if data[subj]['input']['T']['mean'] is not None:
                Tsubj = data[subj]['input']['T']['mean']
            else:
                Tsubj=copy.deepcopy(T)
            avataractivity.add_output(type="constant", name="T_in", f=Tsubj)

            # Set total peripheral resistance (Rtot) based on data
            if 'Rtot' in data[subj]['input'] and data[subj]['input']['Rtot']['mean'] is not None:
                Rtot = data[subj]['input']['Rtot']['mean']
            else:
                # try to load saved values from exercise model simulation
                try:
                    file_name = "./parameters/features_exercisemodel_simulation.json"
                    with open(file_name, "r") as f:
                        loaded_sim = json.load(f)
                    Rtot = loaded_sim[subj][activity]['Rtot']
                except KeyError:
                    print(f'Warning: Rtot not found in data or saved simulation for {subj}, using default value of 1')
                    Rtot = 1 # default value if not found
            if 'currRtot' in avatarmodel.parameter_names:
                params[subj][avatarmodel.parameter_names.index('currRtot')] = Rtot
            avataractivity.add_output(type="constant", name="Rtot_in", f=Rtot)

            # Create simulation objects
            times = data[subj]['input']['simulationtime']
            sims[subj] = sund.Simulation(models = avatarmodel, activities=avataractivity,time_unit = 's',time_vector=times,augment_time_vector=False) 
            
            starttime=0
            simsSS[subj] = sund.Simulation(models = avatarmodel, activities=avataractivity,time_unit = 's',time_vector = np.arange(starttime,starttime+Tsubj*20+0.01,0.01),augment_time_vector=False) 

            # Set simulation settings
            sims[subj].set_options({'abs_tol': 1e-15}) #ami 1e-16, mat 1e-4 , default 1e-06
            sims[subj].set_options({'rel_tol':1e-08}) #ami 1e-8, mat 0.5e-3 , default 1e-06
            sims[subj].set_options({'max_hnil':0}) 
            simsSS[subj].set_options({'abs_tol': 1e-15}) #ami 1e-16, mat 1e-4 , default 1e-06
            simsSS[subj].set_options({'rel_tol':1e-08}) #ami 1e-8, mat 0.5e-3 , default 1e-06
            simsSS[subj].set_options({'max_hnil':0}) 
            
    return avatarmodel, sims, params, simsSS


def setup_avatar_exercise_simulation(modelname,data,DOPPL,opt_exps=['Rest','Exercise','Rampdown','Recovery','RecoveryEnd']):
    sims = {}
    simsSS = {}
    default_params = {}
    for subj in data:
        sims[subj] = {}
        simsSS[subj] = {}
        subj_exps = list(data[subj].keys())
        for exp in subj_exps:
            if exp == 'Rest' or (exp in opt_exps and 'T' in data[subj][exp]['input'] and data[subj][exp]['input']['T']['mean'] is not None):
                model, simtemp, df_par, simSStemp = setup_avatarsimulation_data(modelname,data={subj:data[subj][exp]},activity=exp) #avatar_exercise
                if exp == 'Rest':
                    default_params[subj] = df_par[subj] # use same parameters in all experiments
                if exp in opt_exps:
                    sims[subj][exp] = simtemp[subj]
                    simsSS[subj][exp] = simSStemp[subj]
                elif exp == 'Rest':
                    data[subj].pop(exp)
            else:
                data[subj].pop(exp) # remove the experiment if T is None, so it is not used in optimization

    return sims, simsSS, default_params,model


########################################################################################################################
######################### SIMULATION FUNCTIONS #########################################################################
########################################################################################################################
def exercise_simulation(params, sim, simSS, timepoints=[]):
    # Set initial conditions of the simulation based on parameter values
    if simSS is not None:
        ic=set_IC(params,simSS) 
        # Simulate to steady state 
        simSS.simulate(parameter_values = params, state_values=ic,iterative=True)
        ic=simSS.state_values
    else:
        ic=set_IC(params,sim)

    # Simulate the model
    if len(timepoints)>0:
        sim.simulate(parameter_values = params, state_values=ic,iterative=True,
                     time_vector=timepoints)
    else:
        sim.simulate(parameter_values = params, state_values=ic,iterative=True)

    del ic # clear initial conditions to save memory


def avatar_simulation(params,sim,simSS,timepoints=[]):
    # set subject specific volumes based on total blood volume # alternative: send in the ic to the sim function. possible faster?
    simSS.reset_states() #reset to values in the model
    ic = simSS.state_values
    set_icvolumes(ic,params,simSS) # values in ic are set to values of params for certain state names in simSS

    # %% Simulate to steady state (20 heartbeats)
    #set t0 to start simulation time
    starttime = simSS.time_vector[0]
    params[simSS.parameter_names.index('t0')] = starttime
    #reset sim state values to the ic
    simSS.simulate(parameter_values = params,state_values=ic,iterative=True)
    paramsSS = simSS.parameter_values # to make sure that currT is up to date

    # %% Continue on the previous simulation and simulate the wanted times
    if len(timepoints)>0:
        paramsSS[sim.parameter_names.index('t0')] = timepoints[0]
        sim.simulate(parameter_values = paramsSS,state_values=simSS.state_values,iterative=True,
                     time_vector=timepoints)
    else:
        paramsSS[sim.parameter_names.index('t0')] = sim.time_vector[0] 
        sim.simulate(parameter_values = paramsSS,state_values=simSS.state_values,iterative=True)
    
    del ic # clear initial conditions to save memory

########################################################################################################################
######################### HELPER FUNCTIONS #############################################################################
########################################################################################################################

def set_IC_params(params,data,parameter_names,modelname=None):
    from functions.utils import get_BSA
    #set initial conditions according to data
    paramstoset = ('IC_SV','IC_HR','IC_DBP','IC_SBP','IC_ESV','IC_EDV')
    datanames = ('SV','HR','DBP','SBP','ESV','EDV')
    default = [70, 80, 70, 120,60,130]
    if 'activitystarttime' in data['input']:
        exercisestarttime = data['input']['activitystarttime']
    else:
        exercisestarttime = 0
    for i, p in enumerate(paramstoset):
        if p in parameter_names:
            if datanames[i] in data and data[datanames[i]]['mean'] is not None and data[datanames[i]]['time'][0] <= exercisestarttime:
                ind=np.where(np.array(data[datanames[i]]['time']) <= exercisestarttime)
                params[parameter_names.index(p)] = np.mean(np.array(data[datanames[i]]['mean'])[ind[0]])
            else:
                params[parameter_names.index(p)] = default[i]

    # get BSA from data in case indexed values are given in data
    BSA = get_BSA(data['input'])
    if 'IC_SV' in parameter_names and 'SV' not in data and 'SVi' in data and data['SVi']['mean'] is not None and (data['SVi']['time'][0] <= exercisestarttime):
        params[parameter_names.index('IC_SV')] = data['SVi']['mean'][0]*BSA
    if 'IC_ESV' in parameter_names and 'ESV' not in data and 'ESVi' in data and data['ESVi']['mean'] is not None and (data['ESVi']['time'][0] <= exercisestarttime):
        params[parameter_names.index('IC_ESV')] = data['ESVi']['mean'][0]*BSA
    if 'IC_EDV' in parameter_names and 'EDV' not in data and 'EDVi' in data and data['EDVi']['mean'] is not None and (data['EDVi']['time'][0] <= exercisestarttime):
        params[parameter_names.index('IC_EDV')] = data['EDVi']['mean'][0]*BSA
    
    # if one of edv or esv are given, calculate the other one based on sv:
    if 'IC_SV' in parameter_names:
        SV = params[parameter_names.index('IC_SV')]
    elif 'SV' in data and data['SV']['mean'] is not None and data['SV']['time'][0] <= exercisestarttime:
        SV = data['SV']['mean'][0]
    elif 'SVi' in data and data['SVi']['mean'] is not None and data['SVi']['time'][0] <= exercisestarttime:
        SV = data['SVi']['mean'][0]/BSA
    else:
        SV = None
    if 'IC_ESV' in parameter_names and 'ESV' not in data and 'ESVi' not in data and 'IC_EDV' in parameter_names and SV is not None:
        params[parameter_names.index('IC_ESV')] = params[parameter_names.index('IC_EDV')] - SV
    if 'IC_EDV' in parameter_names and 'EDV' not in data and 'EDVi' not in data and  'IC_ESV' in parameter_names and SV is not None:
        params[parameter_names.index('IC_EDV')] =SV + params[parameter_names.index('IC_ESV')]

   
    # if EDV-ESV and SV data mismatch, add a parameter in the model to correct for that
    if 'SVdiff' in parameter_names:
        if 'SV' in data and data['SV']['mean'] is not None and (data['SV']['time'][0] <= exercisestarttime):
            SVdat = data['SV']['mean'][0]
        elif  'SVi' in data and data['SVi']['mean'] is not None and (data['SVi']['time'][0] <= exercisestarttime):
            SVdat = data['SVi']['mean'][0]*BSA
        else:
            SVdat = None
        
        if 'EDV' in data and data['EDV']['mean'] is not None and (data['EDV']['time'][0] <= exercisestarttime):
            EDVdat = data['EDV']['mean'][0]
        elif 'EDVi' in data and data['EDVi']['mean'] is not None and (data['EDVi']['time'][0] <= exercisestarttime):
            EDVdat = data['EDVi']['mean'][0]*BSA
        else:
            EDVdat = None

        if 'ESV' in data and data['ESV']['mean'] is not None and (data['ESV']['time'][0] <= exercisestarttime):
            ESVdat = data['ESV']['mean'][0]
        elif  'ESVi' in data and data['ESVi']['mean'] is not None and (data['ESVi']['time'][0] <= exercisestarttime):
            ESVdat = data['ESVi']['mean'][0]*BSA
        else:
            ESVdat = None   
        # this is only a problem if all three are given, otherwise we can just use the values as they are
        if SVdat is not None and ESVdat is not None and EDVdat is not None:
            IC_SV= SVdat
            params[parameter_names.index('SVdiff')] = SVdat - (EDVdat-ESVdat)
            # SV=(EDV-ESV) + SVdiff
        else:
            IC_SV = params[parameter_names.index('IC_EDV')] - params[parameter_names.index('IC_ESV')]
            params[parameter_names.index('SVdiff')] = 0
    #otherwise, make sure that EDV, ESV, and SV add upp (SV=EDV-ESV):
    elif 'IC_EDV' in parameter_names and 'IC_ESV' in parameter_names: 
        if 'SV' in data and data['SV']['mean'] is not None and (data['SV']['time'][0] <= exercisestarttime):
            IC_SV = data['SV']['mean'][0]
            svcalc = params[parameter_names.index('IC_EDV')] - params[parameter_names.index('IC_ESV')]
            svdiff = svcalc -IC_SV
            params[parameter_names.index('IC_EDV')] =  params[parameter_names.index('IC_EDV')]-svdiff/2
            params[parameter_names.index('IC_ESV')] =  params[parameter_names.index('IC_ESV')]+svdiff/2
        elif 'SVi' in data and data['SVi']['mean'] is not None and (data['SVi']['time'][0] <= exercisestarttime):
            IC_SV = data['SVi']['mean'][0]*BSA
            svcalc = params[parameter_names.index('IC_EDV')] - params[parameter_names.index('IC_ESV')]
            svdiff = svcalc -IC_SV
            params[parameter_names.index('IC_EDV')] =  params[parameter_names.index('IC_EDV')]-svdiff/2
            params[parameter_names.index('IC_ESV')] =  params[parameter_names.index('IC_ESV')]+svdiff/2
        elif 'IC_SV' in parameter_names:
            IC_SV = params[parameter_names.index('IC_SV')] 
        elif 'IC_EDV' in parameter_names and 'IC_ESV' in parameter_names:
            IC_SV = params[parameter_names.index('IC_EDV')] - params[parameter_names.index('IC_ESV')]

    # set avO2diff initial condition based on VO2 data if it exists and if vo2 data at t <= input time[0], so that we don't take the exercising value as initial condition
    if 'VO2' in data and (data['VO2']['time'][0] <= exercisestarttime): 
        IC_VO2 = data['VO2']['mean'][0]
    elif 'weight' in data['input'] and not isinstance(data['input']['weight'],list) and not data['input']['weight'] is None:
        IC_VO2 = 3*data['input']['weight'] # 2.6-3.5 mL/kg/min
    else:
        IC_VO2 = 3*80

    if 'IC_AvO2_diff' in parameter_names:
        IC_AvO2_diff = IC_VO2/(IC_SV*params[parameter_names.index('IC_HR')]/1000)
        params[parameter_names.index('IC_AvO2_diff')]=IC_AvO2_diff
    if 'IC_VO2' in parameter_names:
        params[parameter_names.index('IC_VO2')]=IC_VO2

    if 'MAP' in data and (data['MAP']['time'][0] <= exercisestarttime) and 'IC_SBP' in parameter_names and 'IC_DBP' in parameter_names:
        # set initial conditions for SBP and DBP based on MAP data
        params[parameter_names.index('IC_SBP')] = 120
        params[parameter_names.index('IC_DBP')] = data['MAP']['mean'][0] - 120*(1/3) # MAP = (1/3)*SBP + (2/3)*DBP

    if 'IC_R' in parameter_names:
        # SVR = (MAP-MRAP)/CO
        if 'SVR' in data and  data['SVR']['mean'] is not None and (data['SVR']['time'][0] <= exercisestarttime):
            IC_R = float(data['SVR']['mean'][0])/80
        elif 'SVR' in data and  data['SVR']['mean'] is not None:
            IC_R = float(np.mean(data['SVR']['mean']))/80   #take mean value if no resting value in order to be in the correct range
        elif 'MAP' in data and (data['MAP']['time'][0] <= exercisestarttime):
            MAP = data['MAP']['mean'][0]
            CO = (IC_SV*params[parameter_names.index('IC_HR')]/1000)
            IC_R = MAP/CO
        else:
            MAP = (1/3)*params[parameter_names.index('IC_SBP')] + (2/3)*params[parameter_names.index('IC_DBP')]
            CO = (IC_SV*params[parameter_names.index('IC_HR')]/1000)
            IC_R = MAP/CO
        params[parameter_names.index('IC_R')] = IC_R

    # set Wmax param if it exists in data, otherwise calculate based on estimated HRmax and measured HRresponse, or last alternative is to use the one given from the params
    if 'Wmax' in parameter_names:
        params[parameter_names.index('Wmax')]=calc_wmax(data)

    params = [np.round(p, ROUNDING_PRECISION) for p in params]


def calc_wmax(data):
    if 'wmax' in data['input'] and data['input']['wmax'] is not None:
        wmax=data['input']['wmax']
    elif ('age' in data['input'] or 'HRmax' in data['input'] or 'HRmax_upright' in data['input']) and 'watt' in data['input'] and data['input']['watt'] is not None and data['HR']['mean'] is not None:
        # assume that % of HR reserve is the same as % of Wmax, and that HRmax = 220-age if not given as an input
        maxwatt = max(data['input']['watt'])
        maxHR = max(data['HR']['mean'])
        HRrest = data['HR']['mean'][0]
        if 'HRmax' in data['input'] and data['input']['HRmax'] is not None:
            HRmax = data['input']['HRmax'] #NOTE: should be max in supine position if data is supine
        elif 'HRmax_upright'  in data['input'] and data['input']['HRmax_upright'] is not None:
            HRmax = data['input']['HRmax_upright']*0.85 #HRmax in supine position is estimated to be 0.85 percent lower than HRmax upright
        else:
            HRmax = 220-data['input']['age']
            HRmax = HRmax*0.85 # specific for supine exercise
        HRmax = max(maxHR,HRmax) # HRmax cannot be smaller than the achieved maximum HR
        percHRreserve = (maxHR-HRrest)/(HRmax-HRrest) # estimation of how close to the individual maximal effort that the performed exercise was
        wmax = maxwatt/percHRreserve #estimation of maximal watt based on the performed watt and the estimated exercise level
    else: # default value
        wmax=200    
    return wmax


def set_IC(params,sim):
    sim.reset_states() #reset to values in the model
    ic = sim.state_values

    if 'bp_exercise_esv' in sim.execution_order():
        IC_SV=(params[sim.parameter_names.index('IC_EDV')]-params[sim.parameter_names.index('IC_ESV')]) + params[sim.parameter_names.index('SVdiff')] 
        IC_AvO2_diff = params[sim.parameter_names.index('IC_VO2')]/(params[sim.parameter_names.index('IC_HR')]*IC_SV/1000)
        ic[sim.state_names.index('AvO2_diff')] = IC_AvO2_diff

    matches = {'SV_ex':'IC_SV','HR':'IC_HR','SBP':'IC_SBP','DBP':'IC_DBP',
               'AvO2_diff':'IC_AvO2_diff','R':'IC_R','VO2':'IC_VO2',
               'EDV':'IC_EDV','ESV':'IC_ESV','SBP_mean':'IC_SBP',
               'DBP_mean':'IC_DBP','HR_mean':'IC_HR','EDV_mean':'IC_EDV',
               'ESV_mean':'IC_ESV','R_mean':'IC_R'}
    for st,par in matches.items():
        if st in sim.state_names and par in sim.parameter_names:
           ic[sim.state_names.index(st)] = params[sim.parameter_names.index(par)]

    return ic

def estimate_sbp(sim,sbp_data):
    '''
    Estimate brachial systolic pressure from simulated aortic pressure to be as close as possible to measured brachial pressure.
    diffparam determining difference between brachial and aortic systolic
    pressure (data from anglo-cardiff trial II http://dx.doi.org/10.1161/HYPERTENSIONAHA.107.105445) 
    8.6 mean value

    DBP is estimated to be the same in the aorta and in the brachial artery, while SBP is estimated to be slightly higher in the brachial artery.
    '''
    if isinstance(sim,sund._Simulation.Simulation):
        idx = sim.feature_names.index('Pao')
        max_pressure = max(sim.feature_data[:, idx])
    else:
        max_pressure = max(sim)
    lb = 0.1
    ub =18.9
    # parameter determining the difference between brachial and aortic SBP
    if sbp_data is None:
        diffparam=0
    else:
        if isinstance(sbp_data,list) or isinstance(sbp_data,np.ndarray):
            sbp_data = sbp_data[0]
        diffparam = max(min(sbp_data-max_pressure,ub),lb)
    return np.array(float(max_pressure + diffparam))



def set_avatarparams(params,parameter_names,onsetLV_sign=None,do_onsetLA=True):
    T=params[parameter_names.index('Torig')]
    # calculate normalization factors for the elastance function
    params[parameter_names.index('norm_factor_LA')]=calculate_norm_factor(T,
                                                        params[parameter_names.index('k_syst_LA')],
                                                        params[parameter_names.index('k_diast_LA')],
                                                        params[parameter_names.index('m1_LA')],
                                                        params[parameter_names.index('m2_LA')])
    params[parameter_names.index('norm_factor_LV')]=calculate_norm_factor(T,
                                                        params[parameter_names.index('k_syst_LV')],
                                                        params[parameter_names.index('k_diast_LV_orig')],
                                                        params[parameter_names.index('m1_LV')],
                                                        params[parameter_names.index('m2_LV')])
    
    #Calculate onset LV (set to +1.5 during optimization to avoid negative values) # if onsetLV_sign=None the given value is used
    if onsetLV_sign == 1:
        params[parameter_names.index('onset_LV')] = params[parameter_names.index('onset_LV')] + 1.5
    elif onsetLV_sign == -1:
        params[parameter_names.index('onset_LV')] = params[parameter_names.index('onset_LV')] - 1.5
    
    #calculate onset LA (LA is optimized to depend on LV)
    if do_onsetLA:
        params[parameter_names.index('onset_LA_orig')] = 1 + params[parameter_names.index('onset_LV')] - params[parameter_names.index('onset_LA_orig')]

def set_exerciseparams(params,parameter_names,rand_sign=None):
    # add before opt, remove before simulation
    randparams = ["keR_r","keSBP_r","keESV_r","k_out_SBP_r","k_out_HR_r","k_out_DBP_r","keDBP_r"]
    for p in randparams:
        if p in parameter_names:
            if rand_sign == 1:
                params[parameter_names.index(p)] =params[parameter_names.index(p)]+0.05
            elif rand_sign == -1:
                params[parameter_names.index(p)] =params[parameter_names.index(p)]-0.05

def load_best_parameters(folder, key=None, param_key='x', cost_key='cost', print_names=False, model=None, direction=1,default_params=None,subject=None):
    """Load the best parameters from a folder with json files. The best parameters are the ones with the lowest cost.
    The function assumes that the json files have the following structure:
    {
        "cost": [cost of the parameters]
        "x": [list of parameters],
    }

    Args:
        folder (string): The folder to search for the results json files.
        key (str, optional): A key that the files must contain to be considered. Defaults to None.
        param_key (str, optional): The key in the json file for parameter values. Defaults to 'x'.
        cost_key (str, optional): The key in the json file for the cost. Defaults to 'cost'.
        print_names (bool, optional): Print the parameter names if True.
        model (sund.model, optional): Model object to load default values from.
        direction (int, optional): Defines if the lowest or greatest solution should be loaded. 

    Returns:
        list: The best parameter values.
    """

    if not os.path.exists(folder) or len(os.listdir(folder))==0:
        print("No best parameters found.")
        if default_params is not None:
            print(f"Using provided default values")
            return copy.deepcopy(default_params)
        elif model is None:
            return []
        else:
            print(f"Using default values from model {model.name}")
            return model.parameter_values
        
    files = os.listdir(folder)
    
    if key is not None:
        files = [f for f in files if key in f] 

    results = []
    for file in files:
        if file.endswith(".json"):
            try:
                with open(f"{folder}/{file}", 'r') as f:
                    res = json.load(f)
                    if ("direction" in res and res["direction"] != direction) or cost_key not in res:
                        continue # skip results that are not in the right direction
                    results.append(res)
            except json.JSONDecodeError:
                print(f"Could not load {folder}/{file}")
            except FileNotFoundError:
                print(f"The file '{folder}/{file}' does not exist. If it is a temporary file, it might have been removed and this error can be ignored.")
            except PermissionError:
                print(f"Permission denied for file '{folder}/{file}'. If it is a temporary file, it might have been removed and this error can be ignored.")

    if len(results) == 0:
        print("No valid results found.")
        if default_params is not None:
            print(f"Using provided default values")
            return copy.deepcopy(default_params)
        elif model is None:
            return []
        else:
            print(f"Using default values from model {model.name}")
            return model.parameter_values
        
    # find the best results loaded from files
    if direction == 1:
        best_result = min(results, key=lambda x: x[cost_key])
    else: # the actual value of the ppl is saved, to if direction =-1, we want the maximum value
        best_result = max(results, key=lambda x: x[cost_key])
    
    if print_names:
        print(best_result)

    if type(best_result[param_key]) is dict and subject is not None and subject in best_result[param_key]:
        return best_result[param_key][subject]

    return best_result[param_key]


def load_best_parameters_all(subjects,basefolder, key=None, param_key='x', cost_key='cost', print_names=False, model=None, direction=1,default_params=None,common_params=None,paramnames=None,subjectloadfrom=None):
    best_params = {}
    for s in subjects:
        folder = f"{basefolder}/{s}"
        if default_params is None:
            df=None
        else:
            df=default_params[s]
        best_params[s] = load_best_parameters(folder, key, param_key, cost_key, print_names, model, direction,default_params=df,subject=s)
    # make sure all common params are the same. base on the first subject.
    if common_params is not None and paramnames is not None:
        if subjectloadfrom is not None and subjectloadfrom in subjects:
            defaultvals = best_params[subjectloadfrom]# use the selected subject's value as common value
            loadedsubj = subjectloadfrom
        else:
            defaultvals = best_params[list(subjects)[0]]# use the first subject's value as common value
            loadedsubj = list(subjects)[0]
        for s in subjects:
            for i,p in enumerate(common_params):
                if isinstance(paramnames,dict):
                    for m in paramnames:
                        if p in paramnames[m]:
                            pidx = paramnames[m].index(p)
                            c = defaultvals[m][pidx]
                            if abs(best_params[s][m][pidx]- c) > 0:
                                print(f"Warning: {p} is not the same for all subjects ({s}{m})! Using value from subject {loadedsubj} for all subjects.")
                            best_params[s][m][pidx] = copy.deepcopy(c)
                else:
                    c = defaultvals[paramnames.index(p)]
                    if abs(best_params[s][paramnames.index(p)]- c) > 0:
                        print(f"Warning: {p} is not the same for all subjects ({s})! Using value from subject {loadedsubj} for all subjects.")
                    best_params[s][paramnames.index(p)] = copy.deepcopy(c)

    return best_params


def bp_punishment_exercise(sim):
    punishment=0
    icsbp=sim.feature_data[:,sim.feature_names.index('SBP')][0]
    icdbp=sim.feature_data[:,sim.feature_names.index('DBP')][0]
    w = sim.feature_data[:,sim.feature_names.index('percWmax')]
    noex = sim.feature_data[np.where(w==0),sim.feature_names.index('SBP')]
    ex = sim.feature_data[np.where(w>0),sim.feature_names.index('SBP')]
    noexD = sim.feature_data[np.where(w==0),sim.feature_names.index('SBP')]
    exD = sim.feature_data[np.where(w>0),sim.feature_names.index('SBP')]
    if sim.feature_data[:,sim.feature_names.index('SBP')].max() > icsbp*2:
        punishment+=100*(sim.feature_data[:,sim.feature_names.index('SBP')].max()-icsbp*2) 
        #penalize unrealistically high SBP (probably comes from fb)
    elif sim.feature_data[:,sim.feature_names.index('SBP')].min() < icsbp*0.5:
        punishment+=100*(icsbp*0.5-sim.feature_data[:,sim.feature_names.index('SBP')].min()) 
        #penalize unrealistically low SBP (probably comes from fb)
    elif sim.feature_data[:,sim.feature_names.index('SBP')].min() < 80:
        punishment+=100*(80-sim.feature_data[:,sim.feature_names.index('SBP')].min()) 
        #penalize too low SBP
    elif sim.feature_data[:,sim.feature_names.index('DBP')].max() > icdbp*2:
        punishment+=100*(sim.feature_data[:,sim.feature_names.index('DBP')].max()-icdbp*2) 
    elif sim.feature_data[:,sim.feature_names.index('DBP')].min() < icdbp*0.5:
        punishment+=100*(icdbp*0.5-sim.feature_data[:,sim.feature_names.index('DBP')].min()) 
    elif sim.feature_data[:,sim.feature_names.index('DBP')].min() < 50:
        punishment+=100*(50-sim.feature_data[:,sim.feature_names.index('DBP')].min()) 
        #penalize too low DBP
    elif np.max(noex)-np.max(ex) > 30:
        punishment+=50*(np.max(noex)-np.max(ex)) 
        #penalize large increase in SBP after exercise
    elif np.max(exD)-np.max(noexD) > 20:
        punishment+=50*(np.max(exD)-np.max(noexD))
    return punishment

def calculate_punishment_avatar(feature_names,features,print_costs):
        adhoc=0
        # punishment if Emax < Emin (non physiological)
        emaxEminPunishment = 0
        if features[0,feature_names.index('Emax_LA')] < features[0,feature_names.index('Emin_LA')]:
                emaxEminPunishment = 300 + (features[0,feature_names.index('Emin_LA')]-features[0,feature_names.index('Emax_LA')])*10000
        if features[0,feature_names.index('Emax_LV')] < features[0,feature_names.index('Emin_LV')]:
            emaxEminPunishment = emaxEminPunishment + 300 + (features[0,feature_names.index('Emin_LV')]-features[0,feature_names.index('Emax_LV')])*10000
        adhoc += emaxEminPunishment
        if print_costs and emaxEminPunishment>0:
            print(f"Emax < Emin punishment: {emaxEminPunishment}")

        # make sure vla is within physiological range
        vlaPunishment = 0
        vla=features[:,feature_names.index('Vla')] 
        if min(vla) < 5:
            vlaPunishment = abs(min(vla))*1000
        if max(vla) > 80:
            vlaPunishment = vlaPunishment + (max(vla)-80)*1000
        adhoc += vlaPunishment
        if print_costs and vlaPunishment>0:
            print(f"Physiological Vla punishment: {vlaPunishment}")

        # punishment if the total resistance is not as the input Rtot (typically from data)
        rtotPunishment = 0
        Rlower = features[0,feature_names.index('Rlower')]# Rlu+Rleg+Rul
        Rbody = features[0,feature_names.index('Rbody')] #Rtot-Rao
        if Rlower < Rbody*1.05:
            rtotPunishment = (Rbody*1.05-Rlower)*10000
        adhoc += rtotPunishment
        if print_costs and rtotPunishment>0:
            print(f"Rtot punishment: {rtotPunishment}")

        # punishment if AV peak is much larger than MV peak flow (unphysiological)
        av_peak = max(features[:,feature_names.index('AV')])
        mv_peak = max(features[:,feature_names.index('MV')])
        if av_peak > mv_peak*1.5:
            adhoc += (av_peak-mv_peak*1.5)*10
            if print_costs and (av_peak-mv_peak*1.5)*10>0:
                print(f"AV peak punishment: {(av_peak-mv_peak*1.5)*10}")
        
        return adhoc


def calculate_adhoc(doadhoc,modelname,k_obs,obs,d,features,idx,adhoc,print_costs):
    weight=1
    if doadhoc and modelname == 'avatar' and (k_obs == 'AV' or  k_obs == 'AA'):
        # weight=2  av and aa carries similar information, so we divide the cost by two to have the same weight as MV flow if they both are present
        if 'AV' in d.keys() and 'AA' in d.keys():
            weight=0.5
        # make sure that there is only one peak
        if k_obs == 'AA':
            if d['input']['tstartDiast_aa']>30:
                ntfs = 100
                weight = weight*0.4 # av has 40 timeframes and we want to have the same weight as MV and AV flow
            else:
                ntfs = 30
            tdiast=int((d['input']['tstartDiast_aa']/ntfs)*len(features[:, idx]))
        else:
            ntfs = 40
            tdiast=int((d['input']['tstartDiast']/ntfs)*len(features[:, idx]))
        highressim=features[0:tdiast, idx]
        pks,_=find_peaks(highressim,prominence=0.1)
        if len(pks) > 1:
            adhoc+= 300
            if print_costs:
                print(f"{k_obs} peak punishment: {300}")
        # if there is positive flow > 10 ml/s when the valve is closed, we add a penalty
        onetf = int(round((1/ntfs)*len(features[:, idx])))
        zerosim=features[tdiast+onetf*2:-onetf*2, idx]
        adhoc += sum(zerosim[zerosim>10])*10
        if print_costs and sum(zerosim[zerosim>10])*10 > 0:
            print(f"{k_obs} zerosim punishment: {sum(zerosim[zerosim>10])*10}")
    elif doadhoc and modelname == 'avatar' and (k_obs == 'MV'):
        # make sure that there is not more than 2 peaks
        ntfs = d['input']['tstartDiast']+len(obs["time"])-1
        tdiast=int((d['input']['tstartDiast']/ntfs)*len(features[:, idx]))
        onetf = int(round((1/ntfs)*len(features[:, idx])))
        highressim=features[tdiast-onetf:, idx]
        pks,_=find_peaks(highressim,prominence=0.1)
        if len(pks) > 2:
            adhoc+= 300
            if print_costs:
                print(f"MV peak punishment: {300}")
        # if there is positive flow > 10 ml/s when the valve is closed, we add a penalty
        zerosim=features[onetf*3:tdiast-onetf*3, idx]
        adhoc += sum(zerosim[zerosim>10])*10
        if print_costs and sum(zerosim[zerosim>10])*10 > 0:
            print(f"{k_obs} zerosim punishment: {sum(zerosim[zerosim>10])*10}")
    elif doadhoc and modelname == 'avatar' and (k_obs == 'DBP' or k_obs == 'SBP' or len(obs['mean']) == 1):
        # if the cost is for DBP or SBP, we want to add weights to the cost
        # 5 times the value to make up for the 20 tfs in the flow data
        weight=5
    return weight,adhoc


# Define cost function
def f_cost_all(params, sims,D,modelname,print_costs=False,doadhoc=False,simsSS=None,do_validation=False,do_plot=False):
    cost = 0
    adhoc = 0
    for k_subj, d in D.items():
        if simsSS is not None:
            simSS=simsSS[k_subj]
        else:
            simSS=None
        c = f_cost(params[k_subj], sims[k_subj], d,modelname,print_costs,doadhoc,simSS=simSS,do_validation=do_validation,do_plot=do_plot)
        if print_costs:
            print(f"Subject {k_subj}: {c}")
        cost += c
    return cost + adhoc

def f_cost(params, sim, d,modelname,print_costs=False,doadhoc=False,simSS=None,do_validation=False,do_plot=False):
    cost = 0
    adhoc = 0
    
    try:
        k_obs = 'simulation'
        if isinstance(sim,dict):
            # for datasets with several experiments using the same parameters
            if modelname == 'avatar_exercise':
                 # Go through rest, exercise etc for each subject
                modelname= 'avatar'
            for exp in sim:
                ss = simSS[exp] if simSS is not None else None
                costexp = f_cost(params, sim[exp], d[exp], modelname, print_costs=print_costs, doadhoc=doadhoc, simSS=ss,do_plot=do_plot)
                cost += costexp
                if print_costs:
                    print(f"{exp}: {costexp} (total so far: {cost})")
            return cost
        elif modelname == 'exercise':
            exercise_simulation(params, sim,simSS)
        elif modelname == 'avatar':
            avatar_simulation(params,sim,simSS)

        feature_names = sim.feature_names
        features = sim.feature_data

        # add punishment for unphysiological behavior
        if modelname == 'avatar':
            adhoc += calculate_punishment_avatar(feature_names,features,print_costs)
        elif modelname == 'exercise':
            bp_punishment = bp_punishment_exercise(sim)
            cost += bp_punishment
        # create plot
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,8))
            pltnum=1
            ncols=2
            nrows = int(np.ceil(len(d.keys())/ncols))
        # calculate cost for each observation in data
        for k_obs, obs in d.items():
            if k_obs == 'SV' and 'SVi' in d.keys():
                continue # skip SV if SVi is available, since it is a duplicate otherwise
            if k_obs == 'EDV' and 'EDVi' in d.keys():
                continue # skip EDV if EDVi is available, since it is a duplicate otherwise
            if k_obs == 'ESV' and 'ESVi' in d.keys():   
                continue # skip ESV if ESVi is available, since it is a duplicate otherwise

            if k_obs not in ["input", "meta", "extra", "all_times"] and obs['mean'] is not None and (not isinstance(obs['mean'],(float,int)) and obs['SEM'][0] != np.inf):
                # remove data points that were used for estimation of IC - only during validation
                if do_validation and k_obs in ['SVi','EDVi','ESVi','SV','EDV','ESV','SVR','TSR']:
                    if obs['mean'] is not None and obs['time'][0] <= d['input']['activitystarttime']:
                        if len(obs['mean']) > 1:
                            obs['mean'] = obs['mean'][1:]
                            obs['time'] = obs['time'][1:]
                            obs['SEM'] = obs['SEM'][1:]
                        else:
                            continue # skip if only the calibration point is given

                # Extract simulation y_sim that matches data
                mask=None
                if modelname == 'avatar' and k_obs == 'SBP':
                    y_sim = estimate_sbp(sim,obs['mean'])                            
                elif modelname == 'avatar' and k_obs in ['DBP','ESV']:
                    names = ['Pao','Vlv']
                    idx = feature_names.index(names[['DBP','ESV'].index(k_obs)])
                    y_sim = min(features[:, idx])
                elif modelname == 'avatar' and k_obs in ['Emax_LV','EDV']:
                    names = ['Elv','Vlv']
                    idx = feature_names.index(names[['Emax_LV','EDV'].index(k_obs)])
                    y_sim = max(features[:, idx])
                else:
                    idx = feature_names.index(k_obs)
                    y_sim = features[:, idx]
                    y_sim = y_sim[np.searchsorted(sim.time_vector, obs["time"])]

                # Calculate cost
                weight,adhoc_calc = calculate_adhoc(doadhoc,modelname,k_obs,obs,d,features,idx,adhoc,print_costs)
                adhoc+=adhoc_calc
                cost += (np.square((obs['mean']-y_sim)/obs['SEM']).sum())*weight

                if print_costs:
                    c = (np.square((obs['mean']-y_sim)/obs['SEM']).sum())*weight
                    print(f"{k_obs}: {c}")

                if do_plot:
                    plt.subplot(nrows,ncols,pltnum)
                    pltnum+=1
                    plt.errorbar(obs['time'], obs['mean'],obs['SEM'], fmt='o', label='data')
                    plt.plot(obs['time'],y_sim, 'x',label='sim')
                    plt.plot(sim.time_vector,features[:, idx], '-',label='full sim')
                    if mask is not None:
                        plt.plot(sim.time_vector[mask],features[mask, idx],'g.',label='avg interval')
                        plt.plot(sim.time_vector,features[:, feature_names.index(k_obs+"_mean")],'k--')
                    plt.title(f"{k_obs} cost: {(np.square((obs['mean']-y_sim)/obs['SEM']).sum())*weight : .3f}")

                if np.isnan(cost):
                    print('nan cost!')

    except Exception as e:
        if "CVODE" not in str(e):
            print(f"Error in {k_obs}: {e} (fcost)")
        cost += 1e20
        return cost

    return cost + adhoc


def set_icvolumes(ic,params,sim):
    # set volumes in the cardiovascular model
    # ventricular, atrial, and aortic volumes can be set directly from data (these parameter values should be set from before)
    if 'Vuart' in sim.state_names:
        # calculate initial conditions for the volumes based on the total blood volume and the fractions of the different compartments
        Vtot = params[sim.parameter_names.index('Vtot')]
        Vaorta = params[sim.parameter_names.index('IC_Vaorta')]
        Vrest = Vtot - Vaorta- params[sim.parameter_names.index('IC_Vla')] -params[sim.parameter_names.index('IC_Vlv')]
        # Distribution of the rest of the total blood volume.The fractions should sum up to 1: 0.12+0.09+0.05+0.27+0.47 = 1
        params[sim.parameter_names.index('IC_Vpulm')] = 0.12*Vrest
        params[sim.parameter_names.index('IC_Vuart')] = 0.09*Vrest
        params[sim.parameter_names.index('IC_Vlart')] = 0.05*Vrest
        params[sim.parameter_names.index('IC_Vlvein')] = 0.27*Vrest
        params[sim.parameter_names.index('IC_Vuvein')] = 0.47*Vrest 

        # set the initial conditions for the volumes
        ic[sim.state_names.index('Vpulm')] = params[sim.parameter_names.index('IC_Vpulm')]
        ic[sim.state_names.index('Vla')] = params[sim.parameter_names.index('IC_Vla')]
        ic[sim.state_names.index('Vlv')] = params[sim.parameter_names.index('IC_Vlv')]
        ic[sim.state_names.index('Vaorta')] = params[sim.parameter_names.index('IC_Vaorta')]
        ic[sim.state_names.index('Vuart')] = params[sim.parameter_names.index('IC_Vuart')]
        ic[sim.state_names.index('Vlart')] = params[sim.parameter_names.index('IC_Vlart')]
        ic[sim.state_names.index('Vlvein')] = params[sim.parameter_names.index('IC_Vlvein')]
        ic[sim.state_names.index('Vuvein')] = params[sim.parameter_names.index('IC_Vuvein')]
        
        # set the rest of the initial conditions
        ic[sim.state_names.index('Qpv')]=0
        ic[sim.state_names.index('Qmv')]=0
        ic[sim.state_names.index('Qav')]=0
        ic[sim.state_names.index('Qaa')]=0
        if 'mv_open' in sim.state_names:
            ic[sim.state_names.index('mv_open')]=1
        if 'av_open' in sim.state_names:
            ic[sim.state_names.index('av_open')]=1
        if 'intQav' in sim.state_names:
            ic[sim.state_names.index('intQav')]=0
        if 'SVav' in sim.state_names:
            ic[sim.state_names.index('SVav')]=0
        if 'EDVprev' in sim.state_names:
            ic[sim.state_names.index('EDVprev')]=0
        if 'ESVprev' in sim.state_names:
            ic[sim.state_names.index('ESVprev')]=10000
        if 'MAP' in sim.state_names:
            ic[sim.state_names.index('MAP')]=100

