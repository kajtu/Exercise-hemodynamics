import copy
import datetime
import json
import os
import sys
import time
from contextlib import contextmanager
from json import JSONEncoder

import numpy as np
from scipy.optimize import Bounds, differential_evolution, dual_annealing
from scipy.stats import qmc

from functions.simfunctions import estimate_sbp,f_cost, f_cost_all, load_best_parameters_all,load_best_parameters,set_avatarparams, set_exerciseparams
from functions.simfunctions import exercise_simulation,avatar_simulation

ROUNDING_PRECISION = 15 # offset pythons default rounding error of 2.2e-16 when exp()


@contextmanager
def silent_errors(stdchannel, dest_filename):
    dest_file = None
    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())
        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


    
def deconstruct_parametervector(param_names,params_allsubjects,data,modelname,optimize_indices_common,optimize_indices_subjspec,constant_indices):
    subj_indices ={}
    theta0_constants={}
    
    # Loop through every subject/dataset
    for i,subj in enumerate(params_allsubjects):
        theta0_subj = params_allsubjects[subj]

        # Add constants for this subject
        theta0_constants[subj] =  list(np.array(theta0_subj)[constant_indices])
        
        # Add common parameters
        if i ==0:
            theta0_common = list(np.array(theta0_subj)[optimize_indices_common])
            theta0 = theta0_common # common params are always first
            subj_indices['common'] = np.arange(len(theta0_common))

            param_names_theta = list([param_names[i] for i in optimize_indices_common])

        # Add subject specific params to the vector of parameters to optimize
        theta0_subjspec = list(np.array(theta0_subj)[optimize_indices_subjspec])
        start_idx = len(theta0)
        theta0.extend(theta0_subjspec)
        end_idx = len(theta0)
        subj_indices[subj] = np.arange(start_idx, end_idx)

        param_names_theta.extend(list([param_names[i] for i in optimize_indices_subjspec]))
        
    return theta0, theta0_constants, subj_indices, param_names_theta

def reconstruct_parametervector(theta0, theta0_constants, param_names,subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices):
        params_allsubjects = {}
        theta0_common = [theta0[i] for i in subj_indices['common']]

        for subj in subj_indices:
            if subj != 'common':
                params_allsubjects[subj] = list(np.zeros(len(theta0_constants[subj])+len(subj_indices[subj])+len(optimize_indices_common)))
                # add common parameters to this subject/dataset
                for idx, val in enumerate(optimize_indices_common):
                    params_allsubjects[subj][val] = theta0_common[idx]
                # add optimized parameters for this subject/dataset
                optparams_subj = [theta0[i] for i in subj_indices[subj]]
                for idx, val in enumerate(optimize_indices_subjspec):
                    params_allsubjects[subj][val]=optparams_subj[idx]
                # add constants for this subject/dataset
                for idx, val in enumerate(constant_indices):
                    params_allsubjects[subj][val]= theta0_constants[subj][idx]
        return params_allsubjects

def f_cost_log(p_log, sims, data, model,theta0_constant=[], constant_indices=[],doadhoc=True):
    p_reconstructed = reconstruct_parametervector(p_log,theta0_constant, constant_indices)
    p = [round(np.exp(x), ROUNDING_PRECISION) for x in p_reconstructed]
    cost = f_cost(p, sims, data, model,doadhoc=doadhoc)
    return cost

def f_cost_log_all(p_log, sims, simsSS,data, modelname,theta0_constant,subj_indices, constant_indices=[],optimize_indices_subjspec=[],optimize_indices_common=[],doadhoc=True):
    if modelname == 'avatar_exercise':
        param_names = sims[next(iter(sims))]['Rest'].parameter_names
    else:
        param_names = sims[next(iter(sims))].parameter_names
    p = [round(np.exp(x), ROUNDING_PRECISION) for x in p_log] #uses relatively lots of memory
    params_allsubjects_reconstructed = reconstruct_parametervector(p, theta0_constant,param_names, subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices)
    
    # %% Set parameter values
    optnames = [p for i,p in enumerate(param_names) if i in optimize_indices_common or i in optimize_indices_subjspec]
    if modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames:
        for subj,sim in sims.items():
            params = params_allsubjects_reconstructed[subj]
            set_avatarparams(params,param_names,onsetLV_sign=-1,do_onsetLA=True)# remove 1.5 
    elif modelname == 'exercise':
        for subj,sim in sims.items():
            params = params_allsubjects_reconstructed[subj]
            # set random effect parameters to the span -0.05-0.05 before simulation
            set_exerciseparams(params,param_names,rand_sign=-1)

    cost = f_cost_all(params_allsubjects_reconstructed, sims, data, modelname,doadhoc=doadhoc,simsSS=simsSS,print_costs=False)
    return cost

def get_parameter_indices_to_optimize(parameternames, fixed_parameters=[],common_parameters=[]):
    constant_indices, optimize_indices, common_indices = [], [],[]
    constant_indices_common, constant_indices_subjspec, optimize_indices_common,optimize_indices_subjspec = [], [],[],[]

    for idx, pName in enumerate(parameternames):
        if pName in fixed_parameters:
            constant_indices.append(idx)
            if pName in common_parameters:
                constant_indices_common.append(idx)
            else:
                constant_indices_subjspec.append(idx)
        else:
            optimize_indices.append(idx)
            if pName in common_parameters:
                optimize_indices_common.append(idx)
            else:
                optimize_indices_subjspec.append(idx)

        if pName in common_parameters:
            common_indices.append(idx)            
            
    return optimize_indices, constant_indices,common_indices,optimize_indices_common,optimize_indices_subjspec

def get_parameter_bounds(theta, param_names_theta, strict_bounds_parameters=[],data=None,subj_inds=None,no_random_effects=[]):
    '''
    theta: vector of default parameter values for all parameters to be optimized (not in log space)
    param_names_theta : names for each parameter in theta
    '''

    # Set general bounds
    lb = [np.log(1.0e-5)]*(len(theta)) 
    ub = [np.log(100)] *(len(theta))

    # %% Assign specific bounds for avatar exercise model
    for p in param_names_theta:
        if p[0:4] == 'khr_':
            indices = [i for i, val in enumerate(param_names_theta) if val == p]
            for ind in indices:
                lb[ind] = np.log(0.001)
                ub[ind] = np.log(5.0) 

    # %% Assign specific bounds for exercise model
    if 'Wmax' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'Wmax']
        for ind in indices:
            if theta[ind] == 0:
                inc = 1e-6
            else:
                inc = theta[ind]*0.25
            lb[ind] = np.log(theta[ind]-inc) 
            ub[ind] = np.log(theta[ind]+inc) 

    if 'vo2_per_w' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'vo2_per_w']
        for ind in indices:
            lb[ind] = np.log(5)
            ub[ind] = np.log(15)

    if 'IC_VO2' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'IC_VO2']
        for ind in indices:
            lb[ind] = np.log(150)
            ub[ind] = np.log(1200)

    if 'k_out_R' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'k_out_R']
        for ind in indices:
            lb[ind] = np.log(0.001) #cannot be too low, then the R does not recover properly after exercice 0.001
            ub[ind] = np.log(50)

    if 'keESV' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'keESV']
        for ind in indices:
            lb[ind] = np.log(0.00001)
            ub[ind] = np.log(30)

    if 'SVlimit' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'SVlimit']
        for ind in indices:
            lb[ind] = np.log(50)
            ub[ind] = np.log(120)

    # random effects for individuals 
    randparams = ["keR_r","keSBP_r","keESV_r","k_out_SBP_r","k_out_HR_r","k_out_DBP_r","keDBP_r"]
    for p in randparams:
        for subj in subj_inds:
            if subj != 'common' and p in param_names_theta:
                subjnames = [param_names_theta[i] for i in subj_inds[subj]]
                pind = subj_inds[subj][subjnames.index(p)]
                if subj in no_random_effects:
                    #no random effects for this subject/dataset, set narrow bounds
                    if theta[pind]==0:
                        lb[pind] = np.log(0.05-1e-5)
                        ub[pind] = np.log(0.05+1e-5)
                    else:
                        lb[pind] = np.log(theta[pind]-1e-6)
                        ub[pind] = np.log(theta[pind]+1e-6)
                else:
                    lb[pind] = np.log(1e-4)
                    ub[pind] = np.log(0.1) #+-5%, corrected before sim but cannot opt negative values

    # data-based bounds for exercise model:
    # sv ~7+-20 ml, HR 70+-10 10.1016/j.echo.2021.05.012
    paramstoset = ('IC_SV','IC_HR','IC_DBP','IC_SBP','IC_VO2','IC_R','IC_ESV','IC_EDV')
    datanames = ('SV','HR','DBP','SBP','VO2','R','ESV','EDV')
    lbdefault = (50,60,60,110,150,5,20,80)
    ubdefault = (90,80,80,130,1200,30,60,200)
    if data is not None and subj_inds is not None:
        for i,p in enumerate(paramstoset):
            if p in param_names_theta:
                for subj in subj_inds:
                    if subj != 'common':
                        subjnames = [param_names_theta[i] for i in subj_inds[subj]]
                        pind = subj_inds[subj][subjnames.index(p)]
                        dn=datanames[i]
                        if subj not in data:
                            print('Warning: subject '+subj+' not in data, setting narrow bounds for '+p)
                            if theta[pind]==0:
                                lb[pind] = np.log(1e-6)
                                ub[pind] = np.log(2e-6)
                            else:
                                lb[pind] = np.log(theta[pind]-1e-6)
                                ub[pind] = np.log(theta[pind]+1e-6)
                            continue
                        elif 'Full' in data[subj]:
                            datasubj = data[subj]['Full']
                        elif 'Pre' in data[subj]:
                            datasubj = data[subj]['Pre']['Rest']
                        elif 'input' not in data[subj]:
                            datasubj = data[subj][list(data[subj].keys())[0]]
                        else:
                            datasubj =data[subj]

                        if 'activitystarttime' in datasubj['input']:
                            exercisestarttime = datasubj['input']['activitystarttime']
                        else:
                            exercisestarttime = 0
                        if dn in datasubj and datasubj[dn]['mean'] is not None and (datasubj[dn]['time'][0] <= exercisestarttime):
                            lb[pind] = np.log(theta[pind]*0.9)
                            ub[pind] = np.log(theta[pind]*1.1)
                        #resistance can be estimated from CO and MAP (although more uncertain than other data values)
                        elif p == 'IC_R': 
                            if ('SVR' in datasubj and datasubj['SVR']['mean'] is not None and (datasubj['SVR']['time'][0] <= exercisestarttime)):
                                lb[pind] = np.log(theta[pind]*0.8)
                                ub[pind] = np.log(theta[pind]*1.2)
                            else: #default value, use broader bounds
                                lb[pind] = np.log(lbdefault[i])
                                ub[pind] = np.log(ubdefault[i])
                        # indexed values can be converted to absolut values using BSA
                        elif p == 'IC_ESV' and 'ESVi' in datasubj and datasubj['ESVi']['mean'] is not None and (datasubj['ESVi']['time'][0] <= exercisestarttime):
                            lb[pind] = np.log(theta[pind]*0.95)
                            ub[pind] = np.log(theta[pind]*1.05)
                        elif p == 'IC_EDV' and 'EDVi' in datasubj and datasubj['EDVi']['mean'] is not None and (datasubj['EDVi']['time'][0] <= exercisestarttime):
                            lb[pind] = np.log(theta[pind]*0.95)
                            ub[pind] = np.log(theta[pind]*1.05)
                        elif p == 'IC_SV' and 'SVi' in datasubj and datasubj['SVi']['mean'] is not None and (datasubj['SVi']['time'][0] <= exercisestarttime):
                            lb[pind] = np.log(theta[pind]*0.95)
                            ub[pind] = np.log(theta[pind]*1.05)
                        else: #default value, use broader bounds
                            lb[pind] = np.log(lbdefault[i])
                            ub[pind] = np.log(ubdefault[i])



    # %% Assign specific bounds for avatar model
    littVals = [0.94,1.48,4.7945,0.1, 0.170,3,0.080,0.080,0.00050,0.0004,
                0.00020,7.40,0.040,0.0037510,0.18,0.45200,0.110,0.2690,
                1.320,1.32,13.1,27.4,0.15,0.001,4,0.01,0.002,0.00050]
    littnames = ('Rtot', 'Ctot', 'ELCo', 'Caa', 'Emax_LA' ,'Emax_LV_orig',
                  'Emin_LA', 'Emin_LV', 'Lao', 'Lav', 'Lmv', 'Ppu', 
                  'Rao' ,'Rmv', 'k_diast_LA', 'k_diast_LV_orig', 'k_syst_LA',
                    'k_syst_LV' ,'m1_LA', 'm1_LV', 'm2_LA' ,'m2_LV', 'onset_LA_orig',
                    'onset_LV','Cpvc','Rpu','Rpv','Lpv')

    # onset La & Lv set so that they are physiologically reasonable close
    if 'onset_LA_orig' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'onset_LA_orig']
        for ind in indices:
            lb[ind] = np.log(0.12)
            ub[ind] = np.log(0.25) 
    if 'onset_LA' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'onset_LA']
        for ind in indices:
            lb[ind] = np.log(0.12)
            ub[ind] = np.log(0.25) 
    if 'onset_LV' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'onset_LV']
        for ind in indices:
            lb[ind] = np.log(1.4)
            ub[ind] = np.log(1.6) 

    # Factor 5 relative litterature for resistance and k_syst:
    resistances = ('Rtot','Rao','Rmv','Rpu','Rpv','k_syst_LA','k_syst_LV')
    for r in resistances:
        if r in param_names_theta:
            indices = [i for i, val in enumerate(param_names_theta) if val == r]
            for ind in indices:
                lb[ind] = np.log(littVals[littnames.index(r)]/5)
                ub[ind] = np.log(littVals[littnames.index(r)]*5) 
    
    # additional resistances and compliances for legs and right side of circulation
    rightsideR = ('Rul','Rupper','Rleg','Rlu','Raa','Rpvc')
    for r in rightsideR:
        if r in param_names_theta:
            indices = [i for i, val in enumerate(param_names_theta) if val == r]
            for ind in indices:
                lb[ind] = np.log(0.0003)
                ub[ind] = np.log(3) 
    rightsideC = ('Cuart','Clart','Clvein','Cuvein')
    for r in rightsideC:
        if r in param_names_theta:
            indices = [i for i, val in enumerate(param_names_theta) if val == r]
            for ind in indices:
                lb[ind] = np.log(0.01)
                ub[ind] = np.log(30)

    # litterature*6 for inertance:
    intert = ('Lao','Lav','Lmv','Lpv')
    for r in intert:
        if r in param_names_theta:
            indices = [i for i, val in enumerate(param_names_theta) if val == r]
            for ind in indices:
                lb[ind] = np.log(littVals[littnames.index(r)]/6)
                ub[ind] = np.log(littVals[littnames.index(r)]*6)

    # litterature*7 for compliance:
    compl = ('Ctot','Caa','Cpvc','ELCo')
    for r in compl:
        if r in param_names_theta:
            indices = [i for i, val in enumerate(param_names_theta) if val == r]
            for ind in indices:
                lb[ind] = np.log(littVals[littnames.index(r)]/7)
                ub[ind] = np.log(littVals[littnames.index(r)]*7) 
    
    # Factor 3 relative litterature for: (from casas et al)
    compl = ('k_diast_LA','k_diast_LV_orig','k_diast_LV','m1_LA','m1_LV','m2_LA','m2_LV','Ppu')
    for r in compl:
        if r in param_names_theta:
            indices = [i for i, val in enumerate(param_names_theta) if val == r]
            for ind in indices:
                lb[ind] = np.log(littVals[littnames.index(r)]/3)
            
            if r == 'k_diast_LV_orig' or r == 'k_diast_LV':
                indices = [i for i, val in enumerate(param_names_theta) if val == r]
                for ind in indices:
                    ub[ind] = np.log(0.9) 
                #kdiast LV ub set to 0.9 (was unneccesarily wide)
            else:
                indices = [i for i, val in enumerate(param_names_theta) if val == r]
                for ind in indices:
                    ub[ind] = np.log(littVals[littnames.index(r)]*3) 

    # Emax and Emin:
    elast = ('Emax_LA','Emax_LV_orig','Emax_LV','Emin_LA','Emin_LV')
    for r in elast:
        if r in param_names_theta:
            indices = [i for i, val in enumerate(param_names_theta) if val == r]
            for ind in indices:
                ub[ind] = np.log(littVals[littnames.index(r)]*3)
            if r in ('Emax_LV_orig','Emin_LV','Emax_LV'):
                indices = [i for i, val in enumerate(param_names_theta) if val == r]
                for ind in indices:
                    #Emax and Emin LV lb same as before (casas et al 2) (but Emax_LV set from data)
                    lb[ind] = np.log(littVals[littnames.index(r)]/3)
            else:
                indices = [i for i, val in enumerate(param_names_theta) if val == r]
                for ind in indices:
                    # Emax and Emin LA lb lowered to allow for no LA contraction in diseased patients
                    lb[ind] = np.log(littVals[littnames.index(r)]/6)

    # Total blood volume
    if 'Vtot' in param_names_theta:
        indices = [i for i, val in enumerate(param_names_theta) if val == 'Vtot']
        for ind in indices:
            lb[ind] = np.log(theta[ind]*0.95) # 5% variation around default value
            ub[ind] = np.log(theta[ind]*1.05) # 5% variation around default value
    
    # Caa, Aao, EOA_ao, Emax_LV, Ctot and Rtot are calculated from data and
    #  get their bounds from there, but still above 0 for the lower bound  
    multfactor = 3 #+-100% of the data uncertainty (bounds = -2*sd to +2*sd), with a lower limit for lb
    lowervals = [0.005,0.1,0.1,0.1,0.1,0.1]
    dataparams = ('Caa','ELCo','Emax_LV','Emax_LV','Ctot','Rtot')
    paramnames = ('Caa','ELCo','Emax_LV_orig','Emax_LV','Ctot','Rtot')
    if data is not None and subj_inds is not None:
        for i,p in enumerate(paramnames):
            if p in param_names_theta:
                for subj in subj_inds:
                    dn = dataparams[i]
                    if subj != 'common' and dn in data[subj] and data[subj][dn]['SEM'] is not None:
                        dvals =data[subj][dn]
                    elif subj != 'common' and 'Rest' in data[subj] and dn in data[subj]['Rest']:
                        dvals =data[subj]['Rest'][dn]
                    else:
                        dvals = None
                    if dvals is not None and dvals['mean'] is not None and dvals['SEM'] is not None:
                        if isinstance(dvals['SEM'], list):
                            unc=dvals['SEM'][0]
                        else:
                            unc = dvals['SEM']
                        if dvals['mean'] is not None:
                            m = dvals['mean'][0]
                        else:
                            m = theta[param_names_theta.index(p)]
                        subjnames = [param_names_theta[i] for i in subj_inds[subj]]
                        pind = subj_inds[subj][subjnames.index(p)]
                        lb[pind] = np.log(float(max(lowervals[i],m-unc*multfactor)))
                        ub[pind] = np.log(float(m+unc*multfactor))
                        if m < lowervals[i]:
                            print(f"Warning: Data value {m} below lower limit {lowervals[i]} for parameter {p}, lb set to {lowervals[i]}")
                        if lb[pind] >= ub[pind]:
                            print(f"Warning: lb {lb[pind]} is higher than ub {ub[pind]} for parameter {p}, ub increased to {lb[pind] + abs(lb[pind])*0.1}")
                            ub[pind] = lb[pind] + abs(lb[pind])*0.1
    
    #%% Make sure that subjects that are not in the data have narrow bounds (for example validation data during ppl)
    for subj in subj_inds:
        if subj not in data and subj != 'common':
            print('Warning: subject '+subj+' not in data, setting narrow bounds for all subject-specific parameters')
            subjnames = [param_names_theta[i] for i in subj_inds[subj]]
            for p in subjnames:
                if p not in randparams: #random effects already set above
                    pind = subj_inds[subj][subjnames.index(p)]
                    if theta[pind]==0:
                        lb[pind] = np.log(1e-10)
                        ub[pind] = np.log(2e-10)  
                    else:
                        lb[pind] = np.log(theta[pind]-1e-10)
                        ub[pind] = np.log(theta[pind]+1e-10)  

    
    # %% If parameters with strict bounds are provided - lower the allowed parameterspace
    if len(strict_bounds_parameters)>0:
        lb_original = copy.deepcopy(lb)
        ub_original = copy.deepcopy(ub)

        for param_name in strict_bounds_parameters:
            indices = [i for i, val in enumerate(param_names_theta) if val == param_name]
            for ind in indices:
                lb[ind] = np.log(copy.deepcopy(theta[ind])*0.95)
                ub[ind] = np.log(copy.deepcopy(theta[ind])*1.05)

        # Ensure that none of the stricter/fixed bounds are outside the original bounds
        ub = np.minimum(ub, ub_original)
        lb = np.maximum(lb, lb_original)

    # %% Return bounds object
    bounds = Bounds(lb, ub)
    return bounds


def create_init_pop(theta0_log, bounds, popsize):
    init = []
    samples_theta0 = int(popsize*0.1)
    samples_qmc    = int(popsize*0.9)
    
    # add start guesses close to theta0
    while len(init) < samples_theta0:
        theta_guess = np.log(np.exp(theta0_log) * np.random.uniform(0.95, 1.05, len(theta0_log)))
        init.append(np.clip(theta_guess, bounds.lb+1e-6, bounds.ub-1e-6)) # make sure guess is within bounds
    
    # add start guesses sampled within the bounds
    sampler = qmc.LatinHypercube(d=len(theta0_log))
    sample = sampler.random(n=samples_qmc)
    sample_scaled = qmc.scale(sample, bounds.lb+1e-8, bounds.ub-1e-8)
    
    [init.append(sample) for sample in sample_scaled]
    
    return init

def bounds_to_tuples(lb, ub, len_bounds):
    lb = np.broadcast_to(lb, len_bounds)
    ub = np.broadcast_to(ub, len_bounds)

    lb = [float(x) if x > -np.inf else None for x in lb]
    ub = [float(x) if x < np.inf else None for x in ub]

    return list(zip(lb, ub))


def optimize(input_args):
    # set the seed for each process to avoid getting the same random numbers
    np.random.seed(int(time.time()) + os.getpid())

    # unpack input arguments
    sims = input_args["sims"]
    simsSS = input_args["simsSS"]
    data = input_args["data"]
    model = input_args["model"]
    modelname = input_args["modelname"]
    param_names = input_args["param_names"]
    default_params = input_args["default_params"]
    print_iterations = input_args["print_iter"]
    maxiter = input_args["max_iter"]
    strict_bounds_parameters = input_args["strict_params"]
    fixed_parameters = input_args["fixed_params"]
    common_parameters = input_args["common_parameters"]
    doadhoc = input_args["do_adhoc"] #use adhoc terms in cost function when optimizing
    no_random_effects = input_args["no_random_effects"] #subjects qu without random effects (ie not individuals)

    def callback(p_log, convergence):
        nonlocal best_cost
        cost = f_cost_log_all(p_log,sims, simsSS,data, modelname, theta0_constants, subj_indices,constant_indices,optimize_indices_subjspec,optimize_indices_common,doadhoc)
        # if a better solution is found - save temp file
        if cost < best_cost:
            best_cost = cost
            try:
                if modelname == 'avatar_exercise':
                    param_names = sims[next(iter(sims))]['Rest'].parameter_names
                else:
                    param_names = sims[next(iter(sims))].parameter_names
                p = [round(np.exp(x), ROUNDING_PRECISION) for x in p_log] #uses relatively lots of memory
                params_allsubjects_reconstructed = reconstruct_parametervector(p, theta0_constants,param_names, subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices)
                
                if modelname == 'avatar' or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
                    for params in params_allsubjects_reconstructed.values():
                        set_avatarparams(params,param_names,onsetLV_sign=-1,do_onsetLA=True)# remove 1.5 after optimization and calculated onset LA (same as done in log cost function)
                if modelname == 'exercise':
                    for params in params_allsubjects_reconstructed.values():
                        set_exerciseparams(params,param_names,rand_sign=-1)

                # save temporary result
                for subj in data:
                    file_name = f"./parameters/{savename}/{subj}/{savename}{'-'.join(list(sims.keys()))}-temp-{os.getpid()}.json"
                    with open(file_name, "w") as file:
                        out = {"f": cost, "x": params_allsubjects_reconstructed[subj]}
                        json.dump(out, file, cls=NumpyArrayEncoder)
            except PermissionError:
                print(f"Permission denied for file './parameters/{model.name}/{model.name}-{'-'.join(list(sims.keys()))}-temp-{os.getpid()}.json'. Not saving current solution.")

    def callback_dual_annealing(p_log, cost, context):
        nonlocal n_iter # To access and be able to update the iteration number
        nonlocal best_cost
        if print_iterations:
            print(f"dual_annealing step {n_iter}: f(x) = {cost} - pid= {os.getpid()}")
        n_iter+=1
        # if a better solution is found - save temp file
        if cost < best_cost:
            best_cost = cost
            try:
                if modelname == 'avatar_exercise':
                    param_names = sims[next(iter(sims))]['Rest'].parameter_names
                else:
                    param_names = sims[next(iter(sims))].parameter_names
                p = [round(np.exp(x), ROUNDING_PRECISION) for x in p_log] #uses relatively lots of memory
                params_allsubjects_reconstructed = reconstruct_parametervector(p, theta0_constants,param_names, subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices)
                
                if modelname == 'avatar' or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
                    for params in params_allsubjects_reconstructed.values():
                        set_avatarparams(params,param_names,onsetLV_sign=-1,do_onsetLA=True)# remove 1.5 after optimization and calculated onset LA (same as done in log cost function)
                if modelname == 'exercise':
                    for params in params_allsubjects_reconstructed.values():
                        set_exerciseparams(params,param_names,rand_sign=-1)

                # save temporary result
                for subj in data:
                    file_name = f"./parameters/{savename}/{subj}/{savename}{'-'.join(list(sims.keys()))}-temp-{os.getpid()}.json"
                    with open(file_name, "w") as file:
                        out = {"f": cost, "x": params_allsubjects_reconstructed[subj]}
                        json.dump(out, file, cls=NumpyArrayEncoder)

            except PermissionError:
                print(f"Permission denied for file './parameters/{model.name}/{model.name}-{'-'.join(list(sims.keys()))}-temp-{os.getpid()}.json'. Not saving current solution.")

    optimize_indices, constant_indices,common_indices,optimize_indices_common,optimize_indices_subjspec = get_parameter_indices_to_optimize(param_names, fixed_parameters,common_parameters)
    
    # divide start guess in a part to optimize and a part to remain constant
    # and for each subject, repeat parameters that are patient specific

    # Load the latest best parameters
    subjects = list(data.keys())
    if modelname == 'avatar_exercise':
        params_allsubjects = load_best_parameters_all(subjects,f"./parameters/{modelname}", cost_key='f', model=model,default_params=default_params,common_params=common_parameters,paramnames=param_names)
    else:
        params_allsubjects = load_best_parameters_all(subjects,f"./parameters/{model.name}", cost_key='f', model=model,default_params=default_params,common_params=common_parameters,paramnames=param_names)

    optnames = [p for i,p in enumerate(param_names) if i in optimize_indices]
    if modelname == 'avatar' or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
        for params in params_allsubjects.values():
            set_avatarparams(params,model.parameter_names,onsetLV_sign=0,do_onsetLA=True)# set onset LA based on LV (reverse since this is set inside log cost func later)
            set_avatarparams(params,model.parameter_names,onsetLV_sign=1,do_onsetLA=False)# add 1.5 to onset LV during opt

    if modelname == 'exercise':
        for params in params_allsubjects.values():
            # add 0.05 for optimization of random effect parameters
            set_exerciseparams(params,model.parameter_names,rand_sign=1)

    # get a vector of all parameters to optimize and one dict with a vector of all constant parameters for each subject
    theta0,theta0_constants,subj_indices,param_names_theta = deconstruct_parametervector(param_names,params_allsubjects,data,modelname,optimize_indices_common,optimize_indices_subjspec,constant_indices)
    # to avoid problems with log(0)
    for i,p in enumerate(theta0):
        if p == 0:
            theta0[i] = 1e-10

    theta0_log = np.log(theta0)
    cost_before_opt_log = f_cost_log_all(theta0_log,sims, simsSS,data, modelname, theta0_constants, subj_indices,constant_indices,optimize_indices_subjspec,optimize_indices_common,doadhoc)
    best_cost = cost_before_opt_log
    print(f'optimize: Initial cost before optimization: {cost_before_opt_log} - pid {os.getpid()}')

    # set up param bounds based on the default parameter values based on the data
    theta0_default, *_ = deconstruct_parametervector(param_names,default_params,data,model,optimize_indices_common,optimize_indices_subjspec,constant_indices)
    bounds = get_parameter_bounds(theta0_default,param_names_theta, strict_bounds_parameters,data=data,subj_inds=subj_indices,no_random_effects=no_random_effects)

    # if pertubation of startguess should occur - allow a change from theta0
    if np.random.random_sample() < input_args["perturbation_ratio"]:
        theta0_log = theta0_log * np.random.uniform(0.95, 1.05, len(theta0_log))

    # make sure the params is within bounds
    for idx,p in enumerate(theta0_log):
        if not p<bounds.ub[idx]:
            print(f'optimize: Warning: {param_names_theta[idx]}: {np.exp(p)} is over ub {np.exp(bounds.ub[idx])}')
        if not p>bounds.lb[idx]:
            print(f'optimize: Warning: {param_names_theta[idx]}: {np.exp(p)} is under lb {np.exp(bounds.lb[idx])}')
        if not bounds.lb[idx] < bounds.ub[idx]:
            print(f'optimize: Warning: {param_names_theta[idx]}: lb {np.exp(bounds.lb[idx])} is larger than ub {np.exp(bounds.ub[idx])}, ub set to be 5 percent higher than lb')
            bounds.ub[idx] = bounds.lb[idx]+abs(bounds.lb[idx]*0.05)
    theta0_log = np.clip(theta0_log, bounds.lb+1e-6, bounds.ub-1e-6)

    # package input to objective function
    args = (sims, simsSS,data, modelname, theta0_constants, subj_indices,constant_indices,optimize_indices_subjspec,optimize_indices_common,doadhoc)

    # create folders to save results in
    if modelname == 'avatar_exercise':
        savename = modelname
    else:
        savename = model.name

    if not os.path.exists(f"./parameters/{savename}"):
            os.makedirs(f"./parameters/{savename}")

    for subj in data:
        if not os.path.exists(f"./parameters/{savename}/{subj}"):
            os.makedirs(f"./parameters/{savename}/{subj}")

    do_differential_evolution = bool(np.random.random_sample() <= input_args["de_ratio"])
    
    if do_differential_evolution:
        print(f'Starting differential_evolution for pid {os.getpid()}, subj: {"-".join(sims.keys())}')
        
        # Differential Evolution allows us to define our own inital population
        init = create_init_pop(theta0_log, bounds, input_args["pop_size"])

        with silent_errors(sys.stderr, os.devnull):
            callback = callback
            res = differential_evolution(func=f_cost_log_all, args=args, bounds=bounds,
                                        x0=theta0_log, disp=print_iterations, maxiter=maxiter, callback=callback, init=init)
        print(f'Done with differential_evolution for pid {os.getpid()}, subj: {"-".join(sims.keys())}')
    else:
        # for dual anneling the bounds needs to be converted to touples (only a problem on HPCs)
        bounds = bounds_to_tuples(bounds.lb, bounds.ub, len(bounds.lb))
        
        print(f'Starting dual_annealing for pid {os.getpid()}, subj: {"-".join(sims.keys())}')
        n_iter = 1 # used for printing the current iteration in the dual_annealing callback
        
        with silent_errors(sys.stderr, os.devnull):
            callback=callback_dual_annealing
            res = dual_annealing(func=f_cost_log_all, args=args, bounds=bounds,
                                x0=theta0_log, maxiter=maxiter, callback=callback)
        print(f'Done with dual_annealing for pid {os.getpid()}, subj: {"-".join(sims.keys())}')
            

    # %% Save the final results
    # all subjects in one file
    params_allsubjects_reconstructed = reconstruct_parametervector(np.exp(res['x']), theta0_constants, param_names, subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices)
    if modelname == 'avatar'  or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
        for params in params_allsubjects_reconstructed.values():
            set_avatarparams(params,param_names,onsetLV_sign=-1,do_onsetLA=True)# remove 1.5 after optimization and calculated onset LA (same as done in log cost function)
    if modelname == 'exercise':
        for params in params_allsubjects_reconstructed.values():
            set_exerciseparams(params,param_names,rand_sign=-1)

    # one file per subject, only save if the solution is better than before opt
    if round(res['fun'],3) < round(cost_before_opt_log,3):
        for subj in data:
            file_name = f"./parameters/{savename}/{subj}/{savename} ({res['fun']}) - {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            with open(file_name, "w") as file:
                out = {"f": res['fun'], "x": params_allsubjects_reconstructed[subj]}
                json.dump(out, file, cls=NumpyArrayEncoder)

    # Remove the temporary file created by callback functions
    for subj in data:
        file_name = f"./parameters/{savename}/{subj}/{savename}{'-'.join(list(sims.keys()))}-temp-{os.getpid()}.json"
        try:
            os.remove(file_name)
        except FileNotFoundError:
            print(f"Could not remove temporary file {file_name}. It might have been removed already.")

################################################################################
########################  Prediction Profile Likelihood  #######################
################################################################################

# a objective function for PPL that returns cost and sim value at a specific time point
def f_cost_PPL(p, sims, D, modelname,exp_to_optimize, obs_to_optimize, t_to_optimize,simsSS=None,print_costs=False):
    if (modelname == 'exercise' and 'c1' in D):
        # when optimizing the patspec parameters, remove the individual pats from the total cost
        dat = copy.deepcopy(D)
        for exp in D:
            if exp[0] == 'c' or exp[0] == 'p':
                dat.pop(exp)
    else:
        dat = D
    cost_all_exps = f_cost_all(p, sims, dat, modelname,doadhoc=False,simsSS=simsSS,print_costs=print_costs)
    if exp_to_optimize in D:
        cost_this_exp = f_cost(p[exp_to_optimize], sims[exp_to_optimize], D[exp_to_optimize], modelname,doadhoc=False,simSS=simsSS[exp_to_optimize],print_costs=print_costs)
    else:
        cost_this_exp = 0
        # the experiment stil has to be simulated to get the PPL value
        if modelname == 'exercise':
            exercise_simulation(p[exp_to_optimize], sims[exp_to_optimize],simsSS[exp_to_optimize])
        elif modelname == 'avatar':
            avatar_simulation(p[exp_to_optimize],sims[exp_to_optimize],simsSS[exp_to_optimize])
    del dat

    # If the cost is too large, return immediately - simulation has crashed
    if cost_all_exps > 1e20 or cost_this_exp > 1e20:
        return 1e20, cost_this_exp, cost_all_exps
    try:
        # Proceed only if the simulation has not crashed
        if modelname == 'avatar_exercise':
            subj = list(D.keys())[0]
            sim = sims[subj][exp_to_optimize]
            simrest = sims[subj]['Rest']
            simex = sims[subj]['Exercise']
            mname='avatar'
        else:
            mname = modelname
            sim = sims[exp_to_optimize]
            if isinstance(sim,dict) and 'input' not in sim:
                simname = list(sim.keys())[0]
                sim = sim[simname]
        # special case for some variables that need to be calculated
        if mname == 'avatar' and obs_to_optimize == 'SBP':
            sbpd = D[exp_to_optimize]['SBP']['mean'] if exp_to_optimize in D and 'SBP' in D[exp_to_optimize] else None
            PPL_value = estimate_sbp(sim,sbpd)                            
        elif mname == 'avatar' and obs_to_optimize == 'DBP':
            idx = sim.feature_names.index('Pao')
            PPL_value = min(sim.feature_data[:, idx])
        elif mname == 'avatar' and obs_to_optimize == 'ESV':
            idx = sim.feature_names.index('Vlv')
            PPL_value = min(sim.feature_data[:, idx])
        elif mname == 'avatar' and obs_to_optimize == 'EDV':
            idx = sim.feature_names.index('Vlv')
            PPL_value = max(sim.feature_data[:, idx])
        elif mname == 'avatar' and obs_to_optimize == 'Emax_LV':
            idx = sim.feature_names.index('Elv')
            PPL_value = max(sim.feature_data[:, idx])
        elif mname == 'avatar' and obs_to_optimize == 'Emax_LA':
            idx = sim.feature_names.index('Ela')
            PPL_value = max(sim.feature_data[:, idx])
        elif mname == 'avatar' and obs_to_optimize == 'LAP':
            idx = sim.feature_names.index('P_LA')
            PPL_value = np.mean(sim.feature_data[:, idx])
        elif mname == 'avatar' and obs_to_optimize == 'LVP':
            idx = sim.feature_names.index('P_LV')
            PPL_value = np.mean(sim.feature_data[:, idx])
        elif modelname == 'avatar_exercise' and obs_to_optimize in ['Emax_LV_change','Emax_LA_change','Rupper_change','Rtot_change','Rbody_change']:
            if obs_to_optimize == 'Emax_LV_change':
                idx = sim.feature_names.index('Elv')
            elif obs_to_optimize == 'Emax_LA_change':
                idx = sim.feature_names.index('Ela')
            elif obs_to_optimize == 'Rupper_change':
                idx = sim.feature_names.index('Rupper')
            elif obs_to_optimize == 'Rtot_change':
                idx = sim.feature_names.index('Rtot')
            elif obs_to_optimize == 'Rbody_change':
                idx = sim.feature_names.index('Rbody')
            rval = max(simrest.feature_data[:, idx])
            eval = max(simex.feature_data[:, idx])
            PPL_value = (eval - rval)/rval*100
        else:
            if modelname == 'vols' and isinstance(sim,dict):
                simname = list(sim.keys())[0]
                idx = sim[simname].feature_names.index(obs_to_optimize)
                tidx = min(np.searchsorted(sim[simname].time_vector, t_to_optimize),len(sim[simname].time_vector)-1)
                PPL_value = sim[simname].feature_data[tidx, idx]
            else:
                idx = sim.feature_names.index(obs_to_optimize)
                tidx =min(np.searchsorted(sim.time_vector, t_to_optimize),len(sim.time_vector)-1)
                PPL_value = sim.feature_data[tidx, idx]
    except Exception as e:
        # Handle any errors and return a large PPL value
        print(f"Error in {exp_to_optimize} {obs_to_optimize} {t_to_optimize}: {e} (fcostppl)")
        PPL_value = 1e20
                    
    return PPL_value, cost_this_exp, cost_all_exps

# objective function for PPL that allows for minimization (direction=1) and maximization (direction=-1)
def f_cost_PPL_log(p_log,sims, simsSS,data, modelname, theta0_constant, subj_indices,constant_indices,optimize_indices_subjspec,optimize_indices_common, exp_to_optimize, obs_to_optimize, t_to_optimize, value_accepted, limit, direction):
    # reconstruct parameter vector from log space into a dictionary of parameters for each subject
    if modelname == 'avatar_exercise':
        param_names = sims[next(iter(sims))]['Rest'].parameter_names
    else:
        param_names = sims[next(iter(sims))].parameter_names
    p = [round(np.exp(x), ROUNDING_PRECISION) for x in p_log] #uses relatively lots of memory
    params_allsubjects_reconstructed = reconstruct_parametervector(p, theta0_constant,param_names, subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices)
    
    # Set parameter values for the avatar model if needed  (contains negative values that need to be reset)
    optnames = [p for i,p in enumerate(param_names) if i in optimize_indices_common or i in optimize_indices_subjspec]
    if modelname == 'avatar' or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
        for subj,sim in sims.items():
            params = params_allsubjects_reconstructed[subj]
            set_avatarparams(params,param_names,onsetLV_sign=-1,do_onsetLA=True)# remove 1.5 
    if modelname == 'exercise':
        for subj,sim in sims.items():
            params = params_allsubjects_reconstructed[subj]
            set_exerciseparams(params,param_names,rand_sign=-1)

    # Calculate the cost and PPL value
    try:
        PPL_value, cost_this_exp, cost_all_exps = f_cost_PPL(params_allsubjects_reconstructed, sims, data, modelname,exp_to_optimize, obs_to_optimize, t_to_optimize,simsSS=simsSS,print_costs=False)
        
        # set PPL objective value (if maximization we make the sim value negative to effectively get the biggest possible "-1*positive" value)
        value = direction*PPL_value 
        
        # if the objective value exceeds the limit - add a punishment
        if cost_this_exp > limit[exp_to_optimize] and cost_all_exps > limit['all']: # it it is not under any limit
            value += (np.abs(value) + np.abs(value_accepted)) * (1 + (cost_this_exp - limit[exp_to_optimize])+ (cost_all_exps - limit['all']))
        elif cost_this_exp > limit[exp_to_optimize]: # if it is not under the experiment speicifc limit
            value += (np.abs(value) + np.abs(value_accepted)) * (1 + (cost_this_exp - limit[exp_to_optimize]))
        elif cost_all_exps > limit['all']: # if it is not under the total limit
            value += (np.abs(value) + np.abs(value_accepted)) * (1 + (cost_all_exps - limit['all']))

    except Exception as e:
        print(f"f_cost_PPL_log error: {e}")
        value = 1e20

    return value


def optimize_PPL(input_args):
    def callback(x, convergence):
        nonlocal best_PPL_cost
        PPL_cost = f_cost_PPL_log(x,sims, simsSS,data, modelname, theta0_constants, subj_indices,constant_indices,optimize_indices_subjspec,optimize_indices_common, exp_to_optimize, obs_to_optimize, t_to_optimize, value_accepted, limit, direction)
        # if a better solution is found - save temp file
        if PPL_cost < best_PPL_cost:
            best_PPL_cost = PPL_cost
            try:
                # reconstruct parameter vector from log space into a dictionary of parameters for each subject
                if modelname == 'avatar_exercise':
                    param_names = sims[next(iter(sims))]['Rest'].parameter_names
                else:
                    param_names = sims[next(iter(sims))].parameter_names
                p = [round(np.exp(j), ROUNDING_PRECISION) for j in x] #uses relatively lots of memory
                params_allsubjects_reconstructed = reconstruct_parametervector(p, theta0_constants,param_names, subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices)
                
                # Set parameter values for the avatar model if needed  (contains negative values that need to be reset)
                optnames = [p for i,p in enumerate(param_names) if i in optimize_indices_common or i in optimize_indices_subjspec]
                if modelname == 'avatar' or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
                    for subj,sim in sims.items():
                        params = params_allsubjects_reconstructed[subj]
                        set_avatarparams(params,param_names,onsetLV_sign=-1,do_onsetLA=True)# remove 1.5 
                if modelname == 'exercise':
                    for subj,sim in sims.items():
                        params = params_allsubjects_reconstructed[subj]
                        set_exerciseparams(params,param_names,rand_sign=-1)

                PPL_value, _, _ = f_cost_PPL(params_allsubjects_reconstructed, sims, data, modelname,exp_to_optimize, obs_to_optimize, t_to_optimize,simsSS=simsSS,print_costs=False)
                # save temporary result
                if direction == 1:
                    minmax='min'
                else:
                    minmax='max'
                with open(f"./results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/{minmax}-temp-{os.getpid()}.json", 'w') as file:
                    out = {"f": PPL_value, "x": params_allsubjects_reconstructed,"direction": direction}
                    json.dump(out, file, cls=NumpyArrayEncoder)
            except PermissionError:
                print(f"Permission denied for file './results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/{minmax}-temp-{os.getpid()}.json'. Not saving current solution.")

    def callback_dual_annealing(x, PPL_cost, context):
        nonlocal n_iter, best_PPL_cost # To access and be able to update the iteration number
        if print_iterations:
            print(f"dual_annealing step {n_iter}: f(x) = {PPL_cost} - pid= {os.getpid()} - {exp_to_optimize} - {obs_to_optimize} - time value: {t_to_optimize}")
        n_iter+=1
        
        # if a better solution is found - save temp file
        if PPL_cost < best_PPL_cost:
            best_PPL_cost = PPL_cost
            try:
                # reconstruct parameter vector from log space into a dictionary of parameters for each subject
                if modelname == 'avatar_exercise':
                    param_names = sims[next(iter(sims))]['Rest'].parameter_names
                else:
                    param_names = sims[next(iter(sims))].parameter_names
                p = [round(np.exp(j), ROUNDING_PRECISION) for j in x] #uses relatively lots of memory
                params_allsubjects_reconstructed = reconstruct_parametervector(p, theta0_constants,param_names, subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices)
                
                # Set parameter values for the avatar model if needed  (contains negative values that need to be reset)
                optnames = [p for i,p in enumerate(param_names) if i in optimize_indices_common or i in optimize_indices_subjspec]
                if modelname == 'avatar'  or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
                    for subj,sim in sims.items():
                        params = params_allsubjects_reconstructed[subj]
                        set_avatarparams(params,param_names,onsetLV_sign=-1,do_onsetLA=True)# remove 1.5 
                if modelname == 'exercise':
                    for subj,sim in sims.items():
                        params = params_allsubjects_reconstructed[subj]
                        set_exerciseparams(params,param_names,rand_sign=-1)

                # save temporary result
                PPL_value, _, _ = f_cost_PPL(params_allsubjects_reconstructed, sims, data, modelname,exp_to_optimize, obs_to_optimize, t_to_optimize,simsSS=simsSS,print_costs=False)
                if direction == 1:
                    minmax='min'
                else:
                    minmax='max'
                with open(f"./results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/{minmax}-temp-{os.getpid()}.json", 'w') as file:
                    out = {"f": PPL_value, "x": params_allsubjects_reconstructed,"direction": direction}
                    json.dump(out, file, cls=NumpyArrayEncoder)
            except PermissionError:
                print(f"Permission denied for file './results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/{minmax}-temp-{os.getpid()}.json'. Not saving current solution.")


    # set the seed for each process to avoid getting the same random numbers
    np.random.seed(int(time.time()) + os.getpid())
    
    # unpack input arguments
    data                     = input_args["data"]
    sims                     = input_args["sims"]
    simsSS                   = input_args["simsSS"]
    model                    = input_args["model"]
    modelname                = input_args["modelname"]
    param_names              = input_args["param_names"]
    default_params           = input_args["default_params"]
    fixed_parameters         = input_args["fixed_params"]
    strict_bounds_parameters = input_args["strict_params"]
    common_parameters        = input_args["common_parameters"]
    print_iterations         = input_args["print_iter"]
    direction                = input_args["direction"]
    limit                    = input_args["chi2_limits"]
    exp_to_optimize          = input_args["exp_to_optimize"]
    obs_to_optimize          = input_args["obs_to_optimize"]
    t_to_optimize            = input_args["t_to_optimize"]
    nreps                    = input_args["n_reps"]
    no_random_effects        = input_args["no_random_effects"] #subjects without random effects (ie not individuals)
    

    if modelname == 'avatar_exercise':
        modelfoldername = 'avatar_exercise/'+list(sims.keys())[0]
    else:
        modelfoldername = model.name

    okcost=False
    for rep in range(nreps):
        # check if the time point needs to be optimized
        try:
            with open(f"./results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/min_max.json", 'r') as f:
                min_max = json.load(f)
            if direction == 1 and min_max['minimized'] == True:
                return
            elif direction == -1 and min_max['maximized'] == True:
                return
        except json.JSONDecodeError:
            # there has been an error while saving the file - restore it and assume no progress has been made
            min_max = {"minimized": False, "maximized": False}
            with open(f"./results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/min_max.json", 'w') as ff:
                json.dump(min_max, ff, cls=NumpyArrayEncoder)
        except Exception as e:
            print(e)
        
        # Find what parameters to optimize
        optimize_indices, constant_indices,common_indices,optimize_indices_common,optimize_indices_subjspec = get_parameter_indices_to_optimize(param_names, fixed_parameters,common_parameters)
        optnames = [p for i,p in enumerate(param_names) if i in optimize_indices]

        # Load the best parameters with respect to minimization/maximization of the observation "obs_to_optimize" at time "t_to_optimize"
        # loads a dictionary with the best parameters for each subject
        
        if modelname == 'avatar_exercise':
            subj = list(sims.keys())[0]
            loadedparams = load_best_parameters(f"./results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}",
                                                cost_key='f', model=model, direction=direction,
                                                 default_params=default_params[subj])
            if not isinstance(loadedparams, dict): # if we got a dictionary it means that we have a solution
                loadedparams = {subj: loadedparams}
            params_allsubjects = copy.deepcopy(loadedparams)
        else:
            loadedparams = load_best_parameters(f"./results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}",
                                                        key=None,param_key='x',cost_key='f',print_names=False, model=model, direction=direction,default_params=default_params,
                                                        subject=None)
            params_allsubjects = copy.deepcopy(loadedparams)
        
        if modelname == 'avatar'  or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
            # reverse the onset LV and LA parameters to be able to optimize them since onset LV can be negative and onset LA is set based on onset LV
            for params in params_allsubjects.values():
                set_avatarparams(params,model.parameter_names,onsetLV_sign=0,do_onsetLA=True)# set onset LA based on LV (reverse since this is set inside log cost func later)
                set_avatarparams(params,model.parameter_names,onsetLV_sign=1,do_onsetLA=False)# add 1.5 to onset LV during opt
        if modelname == 'exercise':
            for params in params_allsubjects.values():
                # add 0.5 for optimization of random effect parameters
                set_exerciseparams(params,param_names,rand_sign=1)

        # get a vector of all parameters to optimize and one dict with a vector of all constant parameters for each subject
        theta0,theta0_constants,subj_indices,param_names_theta = deconstruct_parametervector(param_names,params_allsubjects,data,modelname,optimize_indices_common,optimize_indices_subjspec,constant_indices)
        # make sure theta0 does not contain 0 before logging (this can happen for example if the default parameters are 0)
        theta0 = np.array(theta0)
        theta0[theta0==0] = 1e-12 # to avoid problems with log of 0.

        theta0_log = np.log(theta0)

        # set up parameter bounds based on the default parameter values based on the data
        theta0_default, *_ = deconstruct_parametervector(param_names,default_params,data,model,optimize_indices_common,optimize_indices_subjspec,constant_indices)
        bounds = get_parameter_bounds(theta0_default,param_names_theta, strict_bounds_parameters,data=data,subj_inds=subj_indices,no_random_effects=no_random_effects)

        # only allow the common parameters and the experiment specific parameters to be optimized widely
        for exp in subj_indices:
            if exp != exp_to_optimize and exp != 'common':
                for idx in subj_indices[exp]:
                    if theta0[idx]==0:
                        bounds.lb[idx] = np.log(0.05-1e-4)
                        bounds.ub[idx] = np.log(0.05+1e-4)
                    else:
                        bounds.lb[idx] = np.log(theta0[idx]*0.98 - 1e-8)
                        bounds.ub[idx] = np.log(theta0[idx]*1.02 + 1e-8)

        # make sure that the start guess is within bounds
        for idx,p in enumerate(theta0_log):
            if not p<bounds.ub[idx]:
                print(f'optimize_PPL: Warning: {param_names_theta[idx]}: {np.exp(p)} is over ub {np.exp(bounds.ub[idx])}')
            if not p>bounds.lb[idx]:
                print(f'optimize_PPL: Warning: {param_names_theta[idx]}: {np.exp(p)} is under lb {np.exp(bounds.lb[idx])}')
            if not bounds.lb[idx] < bounds.ub[idx]:
                print(f'optimize_PPL: Warning: {param_names_theta[idx]}: lb {np.exp(bounds.lb[idx])} is larger than ub {np.exp(bounds.ub[idx])}, ub set to be 5 percent higher than lb')
                bounds.ub[idx] = bounds.lb[idx]+abs(bounds.lb[idx]*0.05)
        theta0_log = np.clip(theta0_log, bounds.lb+1e-6, bounds.ub-1e-6)
        
        # simulate the experiment once to be able to extract the sim value
        value_accepted, cost_accepted_this,cost_accepted_all = f_cost_PPL(loadedparams, sims, data, modelname,exp_to_optimize, obs_to_optimize, t_to_optimize,simsSS=simsSS,print_costs=False)
        del loadedparams

        # find the starting PPL value for use in the callback functions
        best_PPL_cost = f_cost_PPL_log(theta0_log,sims, simsSS,data, modelname, theta0_constants, subj_indices,constant_indices,optimize_indices_subjspec,optimize_indices_common, exp_to_optimize, obs_to_optimize, t_to_optimize, value_accepted, limit, direction)
                
        # package arguments that are to be provided to the objective function
        args = (sims, simsSS,data, modelname, theta0_constants, subj_indices,constant_indices,optimize_indices_subjspec,optimize_indices_common, exp_to_optimize, obs_to_optimize, t_to_optimize, value_accepted, limit, direction)

        do_differential_evolution = bool(np.random.random_sample() <= input_args["de_ratio"])
        

        if do_differential_evolution:
            print(f'Starting differential_evolution for pid {os.getpid()} - {exp_to_optimize} - {obs_to_optimize} - time value: {t_to_optimize}')
            
            # Differential evolution allows us to establish our own initial population
            init = create_init_pop(theta0_log, bounds, input_args["pop_size"])
            with silent_errors(sys.stderr, os.devnull):
                res = differential_evolution(
                    func=f_cost_PPL_log, args=args, bounds=bounds,
                    x0=theta0_log, disp=print_iterations, 
                    maxiter=input_args["max_iter"], callback=callback, init=init)
        else:
            print(f'Starting dual_annealing for pid {os.getpid()} - {exp_to_optimize} - {obs_to_optimize} - time value: {t_to_optimize}')
            n_iter = 1 # used for printing the current iteration in the dual_annealing callback
            bounds = bounds_to_tuples(bounds.lb, bounds.ub, len(bounds.lb))
            with silent_errors(sys.stderr, os.devnull):
                try:
                    res = dual_annealing(
                        func=f_cost_PPL_log, args=args, bounds=bounds,
                        x0=theta0_log, maxiter=input_args["max_iter"], 
                        callback=callback_dual_annealing)
                except Exception as e:
                    print(f"Error during dual_annealing: {e}")

        # reconstruct full parameter vector and find the optimized PPL value 
        # reconstruct parameter vector from log space into a dictionary of parameters for each subject
        p = [round(np.exp(j), ROUNDING_PRECISION) for j in res['x']] #uses relatively lots of memory
        p_full = reconstruct_parametervector(p, theta0_constants,param_names, subj_indices,optimize_indices_subjspec,optimize_indices_common,constant_indices)
        # Set parameter values for the avatar model if needed  (contains negative values that need to be reset)
        if modelname == 'avatar' or (modelname == 'avatar_exercise' and 'onset_LV' in optnames and 'onset_LA_orig' in optnames):
            for subj,sim in sims.items():
                set_avatarparams(p_full[subj],param_names,onsetLV_sign=-1,do_onsetLA=True)# remove 1.5    
        if modelname == 'exercise':
            for subj,sim in sims.items():
                set_exerciseparams(p_full[subj],param_names,rand_sign=-1)

        optimized_PPL_value, cost_this_exp, cost_all_exps = f_cost_PPL(p_full, sims, data, modelname,exp_to_optimize, obs_to_optimize, t_to_optimize,simsSS=simsSS,print_costs=False) 
        
        # Save the final results if cost is under both the individual and the overall chi2 limits
        if direction == 1:
            minmax='min'
        else:
            minmax='max'
        if cost_this_exp < limit[exp_to_optimize] and cost_all_exps < limit['all']:
            file_name = f"./results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/optPPL{minmax}({optimized_PPL_value:e})_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            with open(file_name, "w") as file:
                out = {"f": optimized_PPL_value, "x": p_full, "direction":direction}
                json.dump(out, file, cls=NumpyArrayEncoder)
            okcost=True

        # Remove the temporary file created by the callback function
        try:
            os.remove(f"./results_PPL/{modelfoldername}/{minmax}-temp-{os.getpid()}.json")
        except FileNotFoundError:
            print(f"Could not remove temporary file './results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/{minmax}-temp-{os.getpid()}.json'. It might have been removed already.")

    # Update the min/max status in the min_max.json file
    if okcost:
        if direction == 1:
            min_max['minimized'] = True
        else:
            min_max['maximized'] = True
        # Save the min_max status
        with open(f"./results_PPL/{modelfoldername}/{exp_to_optimize}_{obs_to_optimize}_{t_to_optimize}/min_max.json", 'w') as f:
            json.dump(min_max, f, cls=NumpyArrayEncoder)
        print(f'PPL optimization ok cost for pid {os.getpid()} - {exp_to_optimize} - {obs_to_optimize} - t={t_to_optimize}.')
                

