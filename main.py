####################################################################################
##% Main script for the MeChamP project: modelling exercise
## Author: Kajsa Tunedal, kajsa.tunedal@liu.se
####################################################################################

# %% Setup packages 
import matplotlib.pyplot as plt
from functions.simulation_plot_functions import calc_percchange_table, plot_change_between_sims,prediction_max_exercise, table_paramvalues
from functions.data_plot_functions import plot_data_vs_sim_corr, plot_severalfeatures,plot_avatar_all,plot_allinone_avatar
from functions.simfunctions import load_best_parameters_all, setup_avatar_exercise_simulation,setup_exercise_simulation, f_cost_all,f_cost
from functions.optfunctions import optimize, optimize_PPL
from functions.datafunctions import get_dof_subjects,get_dof,load_litterature_data
from functions.utils import create_exercise_prediction, setup_PPL, collect_PPL_uncertainty
import numpy as np
import pandas as pd
from scipy.stats import chi2
import os, sys, copy, random, time,json

#%%###################################################################################
########## Setup function with specific settings for each model ######################
######################################################################################

def setup(MODELNAME,FIXED_PARAMETERS, COMMON_PARAMETERS, STRICT_BOUNDS_PARAMETERS, VALIDATION_DATA,ESTIMATION_DATA,DOPPL):
    avatarmodelname = 'avatar_exercise_legs6'
    do_rand = False
    if MODELNAME == 'bp_exercise_fb':
        exercisemodelname = 'bp_exercise_fb'
    elif MODELNAME == 'bp_exercise_fb_both':
        MODELNAME = 'exercise'
        exercisemodelname = 'bp_exercise_fb_both'
    elif MODELNAME == 'bp_exercise_fb_both_rand':
        MODELNAME = 'exercise'
        exercisemodelname = 'bp_exercise_fb_both_rand'
        do_rand = True
    else:
        do_rand = False
        exercisemodelname = 'bp_exercise_fb_both_rand'

    pred_features = []
    loaded_params = {}
    sims_val=None
    default_params_val=None
    simsSS_val=None
    SUBJECT_SPECIFIC=False
    N_OPTIMIZATIONS=0
    avatarmodel = None
    exercisemodel = None
    ALL_DATA = {}

    # Load data
    with open("./data/MeChaMPdata_anonymized.json", "r") as f:
        data = json.load(f)
    controls = data['controls']
    patients_pre = data['patients_pre']

    if MODELNAME == 'avatar_exercise':
        # estimate parameters, simulate and plot avatar model in rest and exercise (and recovery) for all subjects
        pred_features = ['MV','AV','AA','Pao','Vlv','Vla','Caa','ELCo',
                         'Emax_LV','Emax_LA','Qleg','P_LV','P_LA','m2_LV','k_syst_LV',
                         'onset_LA','k_diast_LV','Rleg','Rupper','Rlower',
                         'k_diast_LV_change','k_syst_LV_change','m2_LV_change',
                         'Caa_change','onset_LA_change','Rleg_change','Rlower_change','Rupper_change',
                         'Rul_change','Rlu_change','Emax_LV_change','Emax_LA_change',
                         'Rbody_change','Rtot_change']
        if PLOT_ENABLED:
            pred_features += ['PV_LV','PV_LA','SBP','DBP','ESV','EDV']

        SUBJECT_SPECIFIC = True
        
        # %% Step 1: simulate and plot avatar model in rest, exercise, and recovery for all subjects
        # use the subjects that have ok exercise costs
        toopt=[]
        unc_ok=[]
        notok_exercisecost = ['p1', 'p2'] # some no exercisedata, some missing MV/Av flow in rest and cannot correctly predict other flow
        no_exercisedata = ['c4', 'c5', 'c2', 'c7', 'c3', 'c11', 'p5', 'p2', 'p9', 'p5', 'p2'] #too large negative MV flow at exercise and/or rest
        if DOPPL or PLOT_ENABLED:
            for subj in controls:
                if subj not in toopt and subj not in notok_exercisecost and subj not in no_exercisedata and subj not in unc_ok:
                    ESTIMATION_DATA.update({subj:controls[subj]})
                    VALIDATION_DATA.update(copy.deepcopy({subj:controls[subj]}))
            for subj in patients_pre:
                if subj not in toopt and subj not in notok_exercisecost and subj not in no_exercisedata and subj not in unc_ok:
                    ESTIMATION_DATA.update({subj:patients_pre[subj]})     
                    VALIDATION_DATA.update(copy.deepcopy({subj:patients_pre[subj]}))
        else:
            for subj in controls:
                if subj not in notok_exercisecost and subj not in no_exercisedata:
                    ESTIMATION_DATA.update({subj:controls[subj]})
                    VALIDATION_DATA.update(copy.deepcopy({subj:controls[subj]}))
            for subj in patients_pre:
                if subj not in notok_exercisecost and subj not in no_exercisedata:
                    ESTIMATION_DATA.update({subj:patients_pre[subj]})  
                    VALIDATION_DATA.update(copy.deepcopy({subj:patients_pre[subj]}))
    
        # Set up avatar model and create simulation objects with activities based on the input from the data
        opt_exps=['Rest','Exercise']
        # if plotting all states: opt_exps=['Rest','Exercise','Rampdown' ,'Recovery']

        # setup removes all data not needed and creates simulation objects only for the wanted subjects and data
        sims, simsSS, default_params,model = setup_avatar_exercise_simulation(avatarmodelname, ESTIMATION_DATA, DOPPL,opt_exps=opt_exps) 

        if PLOT_ENABLED:
            sims_val, simsSS_val, default_params_val,model_val = setup_avatar_exercise_simulation(avatarmodelname, VALIDATION_DATA, DOPPL,opt_exps=['Rampdown' ,'Recovery']) 
        
        # remove everything unneccesary to save memory
        file_name = "./parameters/features_exercisemodel_simulation.json"
        with open(file_name, "r") as f:
            loaded_sim = json.load(f)

        # load UC results
        UC_data = {}
        try:
            if os.path.exists(f"parameters/bp_exercise_fb_both_rand/UC/UC_PPL_bp_exercise_fb_both_rand.json"):
                with open(f"parameters/bp_exercise_fb_both_rand/UC/UC_PPL_bp_exercise_fb_both_rand.json", 'r') as file:
                    UC_data = json.load(file)
        except:
            print("No UC data found")
        
        inputs_to_remove = [ 'SVR','SV','Rtot','BSA']
        for subj in ESTIMATION_DATA:
            for exp in ESTIMATION_DATA[subj]:
                if 'T' in ESTIMATION_DATA[subj][exp]['input'] and 'SEM' in ESTIMATION_DATA[subj][exp]['input']['T']:
                    ESTIMATION_DATA[subj][exp]['input']['T'].pop('SEM')
                if exp != 'Rest' and 'ELCo' in ESTIMATION_DATA[subj][exp]:
                    ESTIMATION_DATA[subj][exp].pop('ELCo')
                for inp in inputs_to_remove:
                    if inp in ESTIMATION_DATA[subj][exp]['input']:
                        ESTIMATION_DATA[subj][exp]['input'].pop(inp)
                # during optimizaton: match edv and esv predicted by exercise model to the ones from the avatar model at exercise
                if exp == 'Exercise':
                    try:
                        ESV = loaded_sim[subj][exp]['ESV']
                        EDV = loaded_sim[subj][exp]['EDV']
                        ESTIMATION_DATA[subj][exp]['EDV'] = {}
                        ESTIMATION_DATA[subj][exp]['ESV'] = {}
                        ESTIMATION_DATA[subj][exp]['ESV']['mean'] =[ESV]
                        ESTIMATION_DATA[subj][exp]['EDV']['mean'] =[EDV]
                        ESTIMATION_DATA[subj][exp]['EDV']['time']=[0]
                        ESTIMATION_DATA[subj][exp]['ESV']['time']=[0]
                        if UC_data and subj in UC_data and 'ESV' in UC_data[subj] and 'EDV' in UC_data[subj]:
                            minvals = UC_data[subj]['ESV']['Min']
                            maxvals = UC_data[subj]['ESV']['Max']
                            ESTIMATION_DATA[subj][exp]['ESV']['SEM'] = [max(max(maxvals)-max(minvals),10)]
                            minvals = UC_data[subj]['EDV']['Min']
                            maxvals = UC_data[subj]['EDV']['Max']
                            ESTIMATION_DATA[subj][exp]['EDV']['SEM'] = [max(max(maxvals)-max(minvals),10)]
                        else:
                            ESTIMATION_DATA[subj][exp]['ESV']['SEM'] =[10]
                            ESTIMATION_DATA[subj][exp]['EDV']['SEM'] = [10]
                    except:
                        print("No previous simulation data found, not adding esv and edv")

        # ---Define which parameters are fixed and should not be optimized---
        # Previously used exercise effect parameters that are not used in the final version of the model:
        FIXED_PARAMETERS += ['khr_m2LA','khr_m1LA','khr_m1LV','khr_EminLA','khr_EminLV','khr_onsetLV']
        FIXED_PARAMETERS+= ["onset_patspec","m2LV_patspec","emaxLV_patspec","emaxLA_patspec","caa_patspec"]

        # Initial volumes determined as fractions of the total volume, and other constants or inputs that should not be optimized:
        FIXED_PARAMETERS += ["V0_LA","V0_LV","Ks_LA","Ks_LV",
                        "rho_blood","norm_factor_LA","norm_factor_LV","currT","t0","Torig",
                        "IC_Vpulm","IC_Vaorta","IC_Vuart","IC_Vlart","IC_Vlvein","IC_Vuvein",
                        "IC_Vlv","IC_Vla"]
        
        param_names = model.parameter_names

        # ---Load optimized parameters for avatar model at rest for all fixed non-common params:---
        resting_params = load_best_parameters_all(ESTIMATION_DATA.keys(),f"./parameters/{model.name}", cost_key='f', model=model,default_params=default_params)
        resting_params_val = load_best_parameters_all(VALIDATION_DATA.keys(),f"./parameters/{model.name}", cost_key='f', model=model,default_params=default_params_val)
        for param in model.parameter_names:
            if param not in COMMON_PARAMETERS and param in FIXED_PARAMETERS: #keep the loaded avatar params for all non-optimized exercise params
                for subj in default_params:
                    default_params[subj][model.parameter_names.index(param)] = resting_params[subj][model.parameter_names.index(param)]
                if VALIDATION_DATA is not None and len(VALIDATION_DATA)>0:
                    for subj in default_params_val:
                        default_params_val[subj][model.parameter_names.index(param)] = resting_params_val[subj][model.parameter_names.index(param)]

    if MODELNAME == 'exercise':
        # Features to calculate uncertainty for
        pred_features = ['HR','SBP','DBP','SVi','EDVi','ESVi',
                         'CO change','SVR','SVR change','SV change',
                         'HR change','SBP change','DBP change',
                         'keESV_r','keR_r']

        # Separate the exercise data
        fulldata_pats_pre = {exp:dat['Full'] for exp,dat in patients_pre.items()}
        fulldata_contr = {exp:dat['Full'] for exp,dat in controls.items()} 

        # Load litterature data
        _, _, datafit, dataval = load_litterature_data(filename='exercisedata_supine.json',validation=['Mayr','Roberts']) #Kindesjo Calbet
        if not do_rand:
            ESTIMATION_DATA.update(datafit)

        # Load average subjects
        with open("data/MeChaMPdata_average_anonymized.json", "r") as f:
            average_data = json.load(f)
        average_pat = average_data['average_pat']
        average_control = average_data['average_control']

        if do_rand:
            ESTIMATION_DATA.update(fulldata_pats_pre)
            ESTIMATION_DATA.update(fulldata_contr)
            if DOPPL:
                if len(fulldata_pats_pre) > 1 and len(fulldata_contr) > 1:
                    ESTIMATION_DATA.update({'average_control':average_control,'average_pat':average_pat})
                ESTIMATION_DATA.update(datafit)
        elif len(fulldata_pats_pre) > 1 and len(fulldata_contr) > 1:
            ESTIMATION_DATA.update({'average_control':average_control,'average_pat':average_pat})

        # Set up for validation
        if PLOT_ENABLED or DOPPL:
            if do_rand and not DOPPL and len(fulldata_pats_pre) > 1 and len(fulldata_contr) > 1:
                VALIDATION_DATA.update({'average_control':average_control})
                VALIDATION_DATA.update({'average_pat':average_pat})

            VALIDATION_DATA.update(dataval)

            # Remove subjects used for parameter estimation
            for subj in ESTIMATION_DATA:
                if subj in VALIDATION_DATA:
                    VALIDATION_DATA.pop(subj)
        if PLOT_ENABLED or DOPPL:
            model_val, sims_val,default_params_val,simsSS_val  = setup_exercise_simulation(exercisemodelname,VALIDATION_DATA)

        # Set up exercise model and create simulation objects with activities based on the input from the data
        model, sims,default_params,simsSS  = setup_exercise_simulation(exercisemodelname,ESTIMATION_DATA)

        # Remove all unnecessary data
        toremove_input = ['sex','weight','height','BMI','hb','N','BSASEM','BSAALL','HRmaxSEM',
                                'HRmaxALL','wattSEM','wattALL','exercisestarttimeSEM','exercisestarttimeALL',
                                'activitystarttimeSEM','activitystarttimeALL','MET','VO2/kg','timeafter','start','VO2max_supine',
                                'VO2max_upright','Hb']
        toremove_features_averagedata = ['COi','CO','HRperc','SViperc','COiperc','SBPperc','DBPperc','SVRperc'] #only used for plotting
        for exp in ESTIMATION_DATA:
                for item in toremove_input:
                    if item in ESTIMATION_DATA[exp]['input']:
                        ESTIMATION_DATA[exp]['input'].pop(item)
                if exp in ['average_pat','average_control']:
                    for item in toremove_features_averagedata:
                        if item in ESTIMATION_DATA[exp]:
                            ESTIMATION_DATA[exp].pop(item)
        for exp in VALIDATION_DATA:
                for item in toremove_input:
                    if item in VALIDATION_DATA[exp]['input']:
                        VALIDATION_DATA[exp]['input'].pop(item)
                if exp in ['average_pat','average_control']:
                    for item in toremove_features_averagedata:
                        if item in VALIDATION_DATA[exp]:
                            VALIDATION_DATA[exp].pop(item)

        # Define which parameters are fixed, which are common among all subjects and which are patient specific
        FIXED_PARAMETERS += ["delayneeded","vo2_per_w","SVdiff"]
        if 'SVlimit' in model.parameter_names:
            COMMON_PARAMETERS.append('SVlimit')

        COMMON_PARAMETERS += ["keHR","k_out_HR","keESV","k_out_ESV","keEDV","k_out_EDV","keSBP","k_out_SBP","keDBP",
                            "k_out_DBP","keR","k_out_R","keavo2","k_out_avo2"]
        
        # Assume no increase in EDV since these are not athletes:
        FIXED_PARAMETERS += ["keEDV","k_out_EDV"]
        for s in default_params:
            default_params[s][model.parameter_names.index('keEDV')] = 0
            default_params[s][model.parameter_names.index('k_out_EDV')] = 0

        randparams = ["keR_r","keESV_r","keSBP_r","k_out_SBP_r","k_out_HR_r","k_out_DBP_r","keDBP_r"]
        randparamsuse = ["keR_r","keESV_r"] # In the final version of the model, only these two random effects were needed to fit to data.
        if do_rand and not DOPPL:
            # Fix all common parameters when optimizing patient-specific random effect parameters
            SUBJECT_SPECIFIC = True
            for param in model.parameter_names:
                if param in COMMON_PARAMETERS:
                    FIXED_PARAMETERS += [param]
                if param not in randparamsuse and param in randparams:
                    FIXED_PARAMETERS += [param]
        elif do_rand and DOPPL:
            # Fix all random effect parameters not needed
            for param in model.parameter_names:
                if param not in randparamsuse and param in randparams:
                    FIXED_PARAMETERS += [param]
        else:
            # If not rand: fix all subject specific random effect parameters
            FIXED_PARAMETERS += randparams

        STRICT_BOUNDS_PARAMETERS += [] 
        param_names = model.parameter_names

    
    # Delete unneccesary variables
    del controls
    del patients_pre

    # Return all needed variables
    return model, sims, default_params, simsSS, sims_val, default_params_val, simsSS_val, param_names, loaded_params, pred_features,SUBJECT_SPECIFIC,N_OPTIMIZATIONS,avatarmodel,exercisemodel


#%%###################################################################################
#######------------------------ MAIN FUNCTION ----------------------------############
######################################################################################

if __name__ == "__main__":
    # Model name. Valid options: 'avatar_exercise','bp_exercise_fb_both','bp_exercise_fb_both_rand'
    backup_model = 'bp_exercise_fb_both_rand'
    if len(sys.argv) > 1:
        MODELNAME = sys.argv[1]
        if MODELNAME not in ['avatar_exercise','bp_exercise_fb_both','bp_exercise_fb_both_rand']:
            print(f"Invalid model name: {MODELNAME}. Valid options are: 'avatar_exercise','bp_exercise_fb_both','bp_exercise_fb_both_rand'. Using default model '{backup_model}'.")
            MODELNAME = backup_model
    else:
        MODELNAME = backup_model # default model if not specified in terminal 

    # Number of optimizatoins to run. If not specified, it is set to None and no optimization is run.
    if len(sys.argv) > 2:
        nopts = sys.argv[2]
    else:
        nopts = None # default to no optimization if not specified in terminal

    # Flag for doing inverse profile likelihood (uncertainty estimation)
    if len(sys.argv) > 3:
        DOPPL = sys.argv[3].lower() == 'doppl' 
    else:
        DOPPL = False # default to not doing uncertainty estimation if not specified in terminal
    
    # %% Setup model and data
    print(f"Running model: {MODELNAME}")
    # ---declare general settings---
    FIXED_PARAMETERS = []
    COMMON_PARAMETERS = []
    STRICT_BOUNDS_PARAMETERS = []
    VALIDATION_DATA = {} # will be set in setup function
    ESTIMATION_DATA = {} # will be set in setup function
    if DOPPL:
        MAXITER = 100
        PERTURBATION_RATIO = 0.5 #if/how often startguess should be changed or not if np.random.random_sample() < perturbation_ratio
        DIFFERENTIAL_EVOLUTION_RATIO = 0.5  #1=only differential evolution, -1=only dual annealing
        if MODELNAME == 'bp_exercise_fb_both_rand':
            # sampling does not work  with narrow bounds, use dual annealing only
            DIFFERENTIAL_EVOLUTION_RATIO=-1 
    else:
        if MODELNAME == 'avatar_exercise':
            MAXITER = 25
            PERTURBATION_RATIO = 1
        elif MODELNAME == 'avatar':
            MAXITER = 50
            PERTURBATION_RATIO = 0.5
        else:
            MAXITER = 100
            PERTURBATION_RATIO = 0.5 #if/how often startguess should be changed or not if np.random.random_sample() < perturbation_ratio (1= only random,-1=only loaded)
        DIFFERENTIAL_EVOLUTION_RATIO = 0.5 #1=only differential evolution, -1=only dual annealing
    INIT_POPSIZE = 100
    DIRECTION = 1
    # Spefific settings for running on hpc cluster or running on local computer. 
    # if on hpc cluster, it is recommended to set PLOT_ENABLED and PRINT_ITERATIONS to False, since plotting and printing slows down the optimization.
    PLOT_ENABLED = True 
    PRINT_ITERATIONS = True
    
    # ---setup model specific settings---
    model, sims, default_params, simsSS, sims_val, default_params_val, simsSS_val, param_names, loaded_params, pred_features,SUBJECT_SPECIFIC,N_OPTIMIZATIONS,avatarmodel,exercisemodel = setup(MODELNAME,FIXED_PARAMETERS, COMMON_PARAMETERS, STRICT_BOUNDS_PARAMETERS, VALIDATION_DATA,ESTIMATION_DATA,DOPPL)
    if MODELNAME in ['bp_exercise_fb_both','bp_exercise_fb_both_rand']:
        MODELNAME = 'exercise'
    print(f'Model setup done for model {MODELNAME}. N_OPTIMIZATIONS: {N_OPTIMIZATIONS}, SUBJECT_SPECIFIC: {SUBJECT_SPECIFIC}, DOPPL: {DOPPL}.')
    if nopts is not None:
        try:
            N_OPTIMIZATIONS = int(nopts)
            print(f"N_OPTIMIZATIONS set to {N_OPTIMIZATIONS} from terminal instead of default {N_OPTIMIZATIONS}.")
        except ValueError:
            print(f"Invalid number of optimizations: {nopts}. Using default N_OPTIMIZATIONS: {N_OPTIMIZATIONS}")
    else:
        PLOT_ENABLED = True 
        PRINT_ITERATIONS = True
    if DOPPL:
        ALL_DATA = copy.deepcopy(ESTIMATION_DATA)
        simsSS_PPL = simsSS # use the same simsSS for PPL as for estimation data
        sims_PPL = sims
        default_params_PPL = default_params
        if not default_params_val is None and MODELNAME != 'avatar_exercise':
            ALL_DATA.update(VALIDATION_DATA)
            default_params_PPL.update(default_params_val) # use the default params for validation data as well
            simsSS_PPL.update(simsSS_val) # add the simsSS for validation data to the PPL simsSS
            sims_PPL.update(sims_val) # add the sims for validation data to the PPL sims

    # ---Calculate chi2 threshold for the data---
    chi2_estimation_dict = {}
    if MODELNAME == 'avatar_exercise':
        chi2_estimation = chi2.ppf(1-0.05, get_dof_subjects(ESTIMATION_DATA))
        chi2_estimation_dict['all'] = chi2_estimation 
        for subj,d in ESTIMATION_DATA.items():
            chi2_estimation_dict[subj] =  chi2.ppf(1-0.05, get_dof_subjects({subj:d})) 
            if DOPPL and MODELNAME == 'avatar_exercise':
                chi2_estimation_dict['Rest']=chi2_estimation_dict[subj]
                chi2_estimation_dict['Exercise']=chi2_estimation_dict[subj]
    else:
        chi2_estimation = chi2.ppf(1-0.05, get_dof(ESTIMATION_DATA))
        if MODELNAME == 'exercise' and DOPPL:
            estdata = {subj: d for subj, d in ESTIMATION_DATA.items() if subj[0] != 'c' and subj[0] != 'p'} #only the average+litterature exps
            chi2_estimation =  chi2.ppf(1-0.05, get_dof(estdata)) 
        else:
            chi2_estimation = chi2.ppf(1-0.05, get_dof(ESTIMATION_DATA))
        chi2_estimation_dict['all'] = chi2_estimation 
        for subj,d in ESTIMATION_DATA.items():
            chi2_estimation_dict[subj] =  chi2.ppf(1-0.05, get_dof({subj:d})) 

    if len(VALIDATION_DATA) > 0:
        if MODELNAME == 'avatar_exercise':
            # c8 craches, so remove it from the validation chi2 calculation
            if 'c8' in VALIDATION_DATA:
                VALIDATION_DATA.pop('c8')
            #chi2_validation = chi2.ppf(1-0.05, get_dof_subjects(VALIDATION_DATA,do_validation=True))
            chi2_validation = chi2.ppf(1-0.05, get_dof_subjects(VALIDATION_DATA))
            chi2_validation_dict = {}
            chi2_validation_dict['all'] = chi2_validation 
            for subj,d in VALIDATION_DATA.items():
                chi2_validation_dict[subj] =  chi2.ppf(1-0.05, get_dof_subjects({subj:d})) 
        else:
            chi2_validation = chi2.ppf(1-0.05, get_dof(VALIDATION_DATA,do_validation=True))
        if DOPPL:
            for subj,d in VALIDATION_DATA.items():
                chi2_estimation_dict[subj] =  chi2.ppf(1-0.05, get_dof({subj:d},do_validation=True))

    # prepare the arguments for the optimization
    def create_task_args(sims, simsSS, data, model, param_names, default_params,chi2_estimation=None,chi2_estimation_dict=None, dir=DIRECTION, k_exp=None, k_obs=None, time=None,do_adhoc=False,no_random_effects=[]):
        """
        Helper function to create task arguments for optimization.
        """
        return {
            "sims": sims,
            "simsSS": simsSS,
            "data": data,
            "model": model,
            "param_names":param_names,
            "default_params":default_params,
            "modelname":MODELNAME,
            "print_iter": PRINT_ITERATIONS,
            "max_iter": MAXITER,
            "perturbation_ratio": PERTURBATION_RATIO,
            "de_ratio": DIFFERENTIAL_EVOLUTION_RATIO,
            "pop_size": INIT_POPSIZE,
            "fixed_params": FIXED_PARAMETERS,
            "strict_params": STRICT_BOUNDS_PARAMETERS,
            "common_parameters":COMMON_PARAMETERS,
            "do_adhoc": do_adhoc,
            "chi2_limit": chi2_estimation,
            "chi2_limits": chi2_estimation_dict,
            "direction": dir, 
            "exp_to_optimize": k_exp,
            "obs_to_optimize": k_obs,
            "t_to_optimize": time,
            "n_reps" : N_OPTIMIZATIONS,
            "no_random_effects": no_random_effects
        }
    
    # Setup
    # define which subjects should NOT have random effects parameters free during optimization
    if N_OPTIMIZATIONS > 0:
        if MODELNAME == 'exercise':
            if DOPPL:
                no_random_effects = [subj for subj in default_params_PPL if subj[0] != 'c' and subj[0] != 'p'] #no random effects for data that are not patients or controls
            else:
                no_random_effects = [subj for subj in default_params if subj[0] != 'c' and subj[0] != 'p'] #no random effects for data that are not patients or controls
        else:
            no_random_effects = []
    # set the seed for each process to avoid getting the same random numbers
    np.random.seed(int(time.time()) + os.getpid())
    if DOPPL and not (SUBJECT_SPECIFIC and not N_OPTIMIZATIONS > 0): #and N_OPTIMIZATIONS > 0:
        validation_experiments=list(VALIDATION_DATA.keys())
        if SUBJECT_SPECIFIC:
            subjects = list(ESTIMATION_DATA.keys())
            random.shuffle(subjects) #so that the order is different on the different threads on NSC
            subj = copy.deepcopy(subjects[0]) # select one subject to optimize
            print('Running PPL for subject:',subj)
            sims_PPL = {subj: sims_PPL[subj]}
            simsSS_PPL = {subj: simsSS_PPL[subj]}
            ALL_DATA = {subj: ALL_DATA[subj]}
            default_params_PPL = {subj: default_params_PPL[subj]}
            del subjects
            chi2_estimation = chi2_estimation_dict[subj]
            if  MODELNAME == 'avatar_exercise':
                 chi2_estimation_dict = {subj:chi2_estimation,'all':chi2_estimation,'Rest':chi2_estimation,'Exercise':chi2_estimation}
            else:
                chi2_estimation_dict = {subj:chi2_estimation,'all':chi2_estimation}
        
        ALL_DATA_PPL, PPL_combinations = setup_PPL(model, MODELNAME,sims_PPL,simsSS_PPL, ALL_DATA, chi2_estimation, 
                                        validation_experiments,default_params_PPL,COMMON_PARAMETERS,pred_features=pred_features)
        shuffled_PPL_items = list(PPL_combinations.items())
        random.shuffle(shuffled_PPL_items)

    if N_OPTIMIZATIONS>0:
        if DOPPL:
            dirs=[-1, 1]
            for _, PPL_opt in shuffled_PPL_items:
                random.shuffle(dirs)        
                for direction in dirs:
                    args = create_task_args(
                        sims_PPL, simsSS_PPL, ALL_DATA_PPL, model, param_names, default_params_PPL, chi2_estimation,chi2_estimation_dict, direction, 
                        PPL_opt["k_exp"], PPL_opt["k_obs"], PPL_opt["time"],do_adhoc=False,no_random_effects=no_random_effects)
                    try:
                        optimize_PPL(args)
                    except Exception as e:
                        print(f"Error occurred during PPL optimization: {e}")
            print("PPL optimizations completed.")
        else:
            do_adhoc=True
            if SUBJECT_SPECIFIC:
                subjects = list(ESTIMATION_DATA.keys())
                random.shuffle(subjects) #so that the order is different on the different threads on NSC
                if not PLOT_ENABLED:
                    subj = copy.deepcopy(subjects[0]) # select one subject to optimize
                    sim = sims[subj]
                    simSS = simsSS[subj]
                    data = copy.deepcopy(ESTIMATION_DATA[subj])
                    default_p = copy.deepcopy(default_params[subj])
                    del ESTIMATION_DATA
                    del sims
                    del simsSS
                    del subjects
                    del default_params
                else:
                    subj = subjects[0] # select one subject to optimize
                    sim = sims[subj]
                    simSS = simsSS[subj]
                    data = ESTIMATION_DATA[subj]
                    default_p = default_params[subj]
                print(f'Starting {N_OPTIMIZATIONS} optimizations for {subj}...')
                args_s=create_task_args({subj:sim}, {subj:simSS}, {subj:data}, model, param_names, {subj:default_p},
                                        do_adhoc=do_adhoc,no_random_effects=no_random_effects)
                for _ in range(N_OPTIMIZATIONS):
                    optimize(args_s)
            else:
                args=create_task_args(sims, simsSS, ESTIMATION_DATA, model, param_names, default_params,do_adhoc=do_adhoc,no_random_effects=no_random_effects)
                for i in range(N_OPTIMIZATIONS):
                    print(f'Starting optimization {i+1} of {N_OPTIMIZATIONS} optimizations...')
                    optimize(args)

    if DOPPL:
        # collect PPL results for plotting

        if MODELNAME == 'exercise':
            # also add prediction simulations that we want uncertainty in
            pred_sims = {}
            activity = {'watt_in':{'values':[0,41,0],'time':[0,8]}} # same watt for all subjects
            pred_sims['samewatt'] = create_exercise_prediction(activity,ESTIMATION_DATA,model)
            activity2 = {'percWmax_in':{'values':[0,55,70,85,100,0],'time':[0,2,4,6,8]}} # ramp maximum exercise test
            pred_sims['rampmax'] = create_exercise_prediction(activity2,ESTIMATION_DATA,model)
        if MODELNAME == 'avatar_exercise':
            SUBJECT_SPECIFIC = False
            for subj in default_params_PPL:
                print(f'collecting PPL results for subject: {subj} (out of {len(default_params_PPL)})')
                validation_experiments=list(VALIDATION_DATA.keys())
                sims_PPL_subj = {subj: sims_PPL[subj]}
                simsSS_PPL_subj = {subj: simsSS_PPL[subj]}
                chi2_estimation = chi2_estimation_dict[subj]
                chi2_estimation_dict_subj = {subj:chi2_estimation,'all':chi2_estimation,'Rest':chi2_estimation,'Exercise':chi2_estimation}
                ALL_DATA_PPL, PPL_combinations = setup_PPL(model, MODELNAME,sims_PPL_subj,simsSS_PPL_subj, {subj: ALL_DATA[subj]}, chi2_estimation, 
                                                validation_experiments,{subj: default_params_PPL[subj]},COMMON_PARAMETERS,pred_features=pred_features)
                # collect ppl
                collect_PPL_uncertainty({subj: default_params_PPL[subj]},model, MODELNAME, {subj:ALL_DATA[subj]}, sims_PPL_subj,sims,
                                        simsSS_PPL_subj,chi2_estimation_dict_subj,pred_features=pred_features,
                                        COMMON_PARAMETERS=COMMON_PARAMETERS,all_data={subj:ALL_DATA_PPL[subj]},
                                        SUBJECT_SPECIFIC=SUBJECT_SPECIFIC,validation_experiments=validation_experiments,
                                        pred_sims=pred_sims if MODELNAME == 'exercise' else None)        
        else:
            collect_PPL_uncertainty(default_params_PPL,model, MODELNAME, ALL_DATA, sims_PPL,sims,
                                    simsSS_PPL,chi2_estimation_dict,pred_features=pred_features,
                                    COMMON_PARAMETERS=COMMON_PARAMETERS,all_data=ALL_DATA_PPL,
                                    SUBJECT_SPECIFIC=SUBJECT_SPECIFIC,validation_experiments=validation_experiments,
                                    pred_sims=pred_sims if MODELNAME == 'exercise' else None)


    #%%###################################################################################
    ######################### Calculate costs ###############################################
    ######################################################################################
    if PLOT_ENABLED:
        subjects = list(ESTIMATION_DATA.keys())
        if MODELNAME == 'avatar_exercise':
            best_params = load_best_parameters_all(subjects,f"./parameters/{MODELNAME}", cost_key='f', model=model,default_params=default_params,paramnames=param_names)
        else:
            best_params = load_best_parameters_all(subjects,f"./parameters/{model.name}", cost_key='f', model=model,default_params=default_params,common_params=COMMON_PARAMETERS,paramnames=model.parameter_names)

        # --- Calculate estimation costs ---
        print("\n##### Estimation costs #####")
        ESTIMATION_DATA_cost = copy.deepcopy(ESTIMATION_DATA)
        # Flag to print all intermediate costs for each subject and experiment.
        printadhoc=False
        printnoadhoc=False
        cost_estimation = f_cost_all(best_params, sims, ESTIMATION_DATA_cost, MODELNAME, print_costs=printnoadhoc, doadhoc=False,simsSS=simsSS)
        print(f"\nEstimation cost: {cost_estimation:.5f} passes the chi2-threshold: {chi2_estimation:.5f} - ({bool(cost_estimation<chi2_estimation)})")

        # Calculate cost and chi2 limit for each experiment/subject, and save in a table
        chi2s = [None]*(len(ESTIMATION_DATA_cost.keys())+1)
        costs = [None]*(len(ESTIMATION_DATA_cost.keys())+1)
        costsadhoc = [None]*(len(ESTIMATION_DATA_cost.keys())+1)
        idx=0
        for k_subj, d in ESTIMATION_DATA_cost.items():
            if MODELNAME == 'avatar_exercise':
                chi2s[idx] = chi2.ppf(1-0.05, get_dof_subjects({k_subj:d}))
            else:
                chi2s[idx] = chi2.ppf(1-0.05, get_dof({k_subj:d}))
            costs[idx] = f_cost(best_params[k_subj], sims[k_subj], d,MODELNAME,simSS=simsSS[k_subj], print_costs=False)
            costsadhoc[idx] = f_cost(best_params[k_subj], sims[k_subj], d,MODELNAME,simSS=simsSS[k_subj], print_costs=False,doadhoc=True)
            idx+=1
        chi2s[idx] = chi2_estimation
        costs[idx] = cost_estimation
        costsadhoc[idx] = f_cost_all(best_params, sims, ESTIMATION_DATA_cost, MODELNAME, print_costs=printadhoc, doadhoc=True,simsSS=simsSS)
        subjects = list(ESTIMATION_DATA_cost.keys())
        subjects.append('Sum of all')
        relative_costdiff = np.divide((np.array(costs)-np.array(chi2s)),np.array(chi2s))
        costdict = {'Subject':subjects,'Cost opt (adhoc)':costsadhoc,'Cost':costs,'Chi2limit':chi2s,'Ok cost?':np.array(costs)<np.array(chi2s),'Rel cost diff':relative_costdiff}
        df=pd.DataFrame(costdict)
        print(df)
        df.to_excel('./tables/cost_table_'+ model.name +'.xlsx')

        # Create a table of parameter values
        pvals_p = [' ']*len(param_names)
        pvals_c = [' ']*len(param_names)
        pvals_other = [' ']*len(param_names)
        for pidx,p in enumerate(param_names):
            pvals_p_all = [best_params[subj][pidx] for subj in best_params if subj[0] == 'p']
            if pvals_p_all:
                pvals_p[pidx] = f'{np.mean(pvals_p_all):,.4f} [{min(pvals_p_all):,.4f}-{max(pvals_p_all):,.4f}]'
            pvals_c_all = [best_params[subj][pidx] for subj in best_params if subj[0] == 'c']
            if pvals_c_all:
                pvals_c[pidx] = f'{np.mean(pvals_c_all):,.4f} [{min(pvals_c_all):,.4f}-{max(pvals_c_all):,.4f}]'
            pvals_other_all = [best_params[subj][pidx] for subj in best_params if subj[0] not in ['p', 'c']]
            if pvals_other_all:
                pvals_other[pidx] = f'{np.mean(pvals_other_all):,.4f} [{min(pvals_other_all):,.4f}-{max(pvals_other_all):,.4f}]'
        paramdict = {'Parameter names':param_names,'Controls (mean [range])':pvals_c,'Subjects with hypertension (mean [range])':pvals_p, 'Others: (mean [range])':pvals_other}
        df=pd.DataFrame(paramdict)
        df.to_excel('./tables/supplementary_parameter_values_'+model.name+'.xlsx')

        # Find the lowest and highest relative cost differences for patients and controls to identify which subjects are the best and worst fitted for plotting
        relcostdiffsp = [c for i,c in enumerate(relative_costdiff) if subjects[i][0] == 'p']
        relcostdiffsc = [c for i,c in enumerate(relative_costdiff) if subjects[i][0] == 'c']
        if len(ESTIMATION_DATA) > 3 and len(relcostdiffsc) > 0 and len(relcostdiffsp) > 0:
            lowest_highest_estimation={subj:ESTIMATION_DATA[subj] for i,subj in enumerate(subjects) if relative_costdiff[i] == min(relcostdiffsp) or relative_costdiff[i] == max(relcostdiffsp) or relative_costdiff[i] == min(relcostdiffsc) or relative_costdiff[i] == max(relcostdiffsc)}
        else:
            lowest_highest_estimation = ESTIMATION_DATA

        # --- Calculate validation costs ---
        if len(VALIDATION_DATA) > 0:
            #use default params for subject specific params, and use best params for all common params
            best_params_val = {}
            best_params_temp = copy.deepcopy(best_params[list(best_params.keys())[0]])
            for exp in default_params_val:
                best_params_val[exp] = copy.deepcopy(default_params_val[exp])
                for param in COMMON_PARAMETERS:
                    best_params_val[exp][model.parameter_names.index(param)] = best_params_temp[model.parameter_names.index(param)]

        # Calculate costs and chi2 limits for validation data, and save in a table
        # NOTE: Validation data for bp_exercise_fb_both_rand is added purely for plotting purposes, not used to calculate costs
        if len(VALIDATION_DATA) > 0 and model.name != 'bp_exercise_fb_both_rand':
            print("\n##### Validation costs #####")
            VALIDATION_DATA_cost = copy.deepcopy(VALIDATION_DATA)
            cost_validation = f_cost_all(best_params_val, sims_val, VALIDATION_DATA_cost, MODELNAME, print_costs=False,simsSS=simsSS_val,doadhoc=False,do_validation=True)
            print(f"Validation cost: {cost_validation:.5f} passes the chi2-threshold: {chi2_validation:.5f} - ({bool(cost_validation<chi2_validation)})\n")

            # Calculate cost and chi2 limit for each experiment/subject, and save in a table
            chi2s = [None]*(len(VALIDATION_DATA_cost.keys())+1)
            costs = [None]*(len(VALIDATION_DATA_cost.keys())+1)
            costsadhoc = [None]*(len(VALIDATION_DATA_cost.keys())+1)
            idx=0
            for k_subj, d in VALIDATION_DATA_cost.items():
                if MODELNAME == 'avatar_exercise':
                    chi2s[idx] = chi2.ppf(1-0.05, get_dof_subjects({k_subj:d},do_validation=True))
                else:
                    chi2s[idx] = chi2.ppf(1-0.05, get_dof({k_subj:d},do_validation=True))
                costs[idx] = f_cost(best_params_val[k_subj], sims_val[k_subj], d,MODELNAME,simSS=simsSS_val[k_subj], print_costs=False,do_validation=True)
                costsadhoc[idx] = f_cost(best_params_val[k_subj], sims_val[k_subj], d,MODELNAME,simSS=simsSS_val[k_subj], print_costs=False,doadhoc=True,do_validation=True)
                idx+=1
            chi2s[idx] = chi2_validation
            costs[idx] = cost_validation
            costsadhoc[idx] = f_cost_all(best_params_val, sims_val, VALIDATION_DATA_cost, MODELNAME, print_costs=False, doadhoc=True,simsSS=simsSS_val,do_validation=True)
            subjects = list(VALIDATION_DATA_cost.keys())
            subjects.append('Sum of all')
            costdict = {'Subject':subjects,'Cost opt (adhoc)':costsadhoc,'Cost':costs,'Chi2limit':chi2s,'Ok cost?':np.array(costs)<np.array(chi2s)}
            df=pd.DataFrame(costdict)
            print(df)
            df.to_excel('./tables/cost_table_val_'+model.name+'.xlsx')
            okcosts = np.array(costs)<np.array(chi2s)
            ok_validation = {subj:VALIDATION_DATA[subj] for subj in subjects if okcosts[subjects.index(subj)] and subj in VALIDATION_DATA}
            not_ok_validation = {subj:VALIDATION_DATA[subj] for subj in subjects if not okcosts[subjects.index(subj)] and subj in VALIDATION_DATA}

            relative_costdiff = np.divide((np.array(costs)-np.array(chi2s)),np.array(chi2s))
            relcostdiffsp = [c for i,c in enumerate(relative_costdiff) if subjects[i][0] == 'p']
            relcostdiffsc = [c for i,c in enumerate(relative_costdiff) if subjects[i][0] == 'c']
            if len(relcostdiffsp) > 2 and len(relcostdiffsc) > 2:
                lowest_highest_validation={subj:VALIDATION_DATA[subj] for i,subj in enumerate(subjects) if relative_costdiff[i] == min(relcostdiffsp) or relative_costdiff[i] == max(relcostdiffsp) or relative_costdiff[i] == min(relcostdiffsc) or relative_costdiff[i] == max(relcostdiffsc)}
            else:
                lowest_highest_validation = VALIDATION_DATA

    #%%###################################################################################
    ######################### Plot results ###############################################
    ######################################################################################
    if PLOT_ENABLED:
        print("\nCreating figures....")
        # Create results figures for the cardiovascular lumped parameter model at selected heartbeats during rest and exercise
        if MODELNAME == 'avatar_exercise':
            # --Plot predicted changes in model features with exercise (manuscript figure 7)--
            features={'EmaxLV':['max','Elv','mmHg/mL'], 'EmaxLA':['max','Ela','mmHg/mL'],
                      'Central MAP':['mean','Pao','mmHg'],'Mean LVP':['mean','P_LV','mmHg'],
                      'Mean LAP':['mean','P_LA','mmHg'],'Max LAP':['max','P_LA','mmHg'],
                      'Mean leg flow':['mean','Qleg','mL/s']}
            uncnames = {'EmaxLV':'Emax_LV_change','EmaxLA':'Emax_LA_change','Caa':'Caa_change','m2LV':'m2_LV_change','onsetLA':'onset_LA_change',
                        'k_diast_LV':'k_diast_LV_change','k_syst_LV':'k_syst_LV_change',
                        'Rleg':'Rleg_change','Rupper':'Rupper_change','Rlower':'Rlower_change',
                        'Rul':'Rul_change','Rlu':'Rlu_change'}
            subjects = [k for k in ESTIMATION_DATA if k[0] == 'c' or k[0] == 'p']
            if len(subjects) > 2 and 'Rest' in ESTIMATION_DATA[subjects[0]] and 'Exercise' in ESTIMATION_DATA[subjects[0]]:
                plot_change_between_sims(features,subjects,sims,expnames=['Rest','Exercise'],figname='Figure7_avatar_rest_vs_exercise',savefig=True)
                plot_change_between_sims(features,subjects,sims,expnames=['Rest','Exercise'],figname='Figure7_avatar_rest_vs_exercise_percent',savefig=True,percent=True,uncnames=uncnames)
                features={'T':['max','currT','s'],
                        'EmaxLV':['max','Elv','mmHg/mL'], 'EmaxLA':['max','Ela','mmHg/mL'],
                        'Caa':['mean','Caa','mmHg/mL'],'m2LV':['mean','m2_LV','-'],
                        'onsetLA':['mean','onset_LA','-'],'k_diast_LV':['mean','k_diast_LV','-'],
                        'k_syst_LV':['mean','k_syst_LV','-'],
                        'Rleg':['mean','Rleg','mmHg*s/mL'],'Rupper':['mean','Rupper','mmHg*s/mL'],
                        'Rlower':['mean','Rlower','mmHg*s/mL'],'Rbody':['mean','Rbody','mmHg*s/mL'],'Rtot':['mean','Rtot','mmHg*s/mL'],
                        'Rul':['mean','Rul','mmHg*s/mL'],'Rlu':['mean','Rlu','mmHg*s/mL']}
                print("\n##### Table of percentage changes with exercise #####")
                calc_percchange_table(features,subjects,sims,expnames=['Rest','Exercise'],savetables=True,uncnames=uncnames)

            # -- Manuscript figure 6: Fit to data for a selected subject + correlations between simulations and data for all subjects --
            # Plot simulations at rest and exercise in same plot for a selected subject (manuscript figure 6)
            selected_subj = 'p8'
            if (selected_subj in ESTIMATION_DATA and 'Recovery' not in ESTIMATION_DATA[selected_subj]):
                selecteddata = {k:ESTIMATION_DATA[selected_subj][k] for k in ESTIMATION_DATA[selected_subj] if k in ['Rest','Exercise']}
                plot_severalfeatures(best_params[selected_subj],'avatar_exercise',model.name,sims[selected_subj],selecteddata, simSS=simsSS[selected_subj],
                    features_to_plot=[['MV'],['AV'],['AA'],['Qleg'],['PV_LA'],['PV_LV'],['Pao'],
                                                ['k_diast_LV','k_syst_LV','onset_LA','m2_LV','Caa'],['Rleg_ex','Rupper','Rlower','Rbody','Rtot'],
                                                ['Ela'],['Elv'],['P_LA'],['P_LV'],],
                    ylabels=['MV flow (mL/s)','AV flow (mL/s)','AA flow (mL/s)','Leg flow (mL/s)','PVla','pvlv','Ao pressure (mmHg)',
                                'Parameters (log)','Resistances (log)',  'LA Elastance (mmHg/mL)','LV Elastance (mmHg/mL)','LA pressure (mmHg)','LV pressure (mmHg)',],
                            savefig=True,figname='Figure6_avatar_ex_'+selected_subj,wide=False,separate_cols=False,allexpsinone=True,
                            testparams=None,ncols=3,figwidth=17,lw=2,subj=selected_subj)

            # Plot correlation between data and simulation for all subjects (manuscript Figure 6)
            # Note that this requires the 'Recovery' and 'Rampdown' in estimation data - change the variable opt_exps in the setup function to include these experiments if you want to plot this figure.
            if selected_subj in ESTIMATION_DATA and 'Recovery' in ESTIMATION_DATA[selected_subj]:
                plot_data_vs_sim_corr(ESTIMATION_DATA,sims,['MV','AV','AA'],MODELNAME,figname='Figure6_correlation_data_vs_sim_avatar_exercise',savefig=True)

            # Fit to data for a selected subject (manuscript figure 6)
            if selected_subj in ESTIMATION_DATA and 'Recovery' not in ESTIMATION_DATA[selected_subj]:
                # no params --> no new sim is done
                plot_avatar_all(params=None,data=ESTIMATION_DATA[selected_subj],sims=sims[selected_subj],simsSS=simsSS[selected_subj],savefigs=True,figname='Figure6_Fit_ex_estimation_'+selected_subj,subjin=selected_subj) # assuming that simulations are already done
                plot_avatar_all(params=None,data=ESTIMATION_DATA[selected_subj],sims=sims[selected_subj],simsSS=simsSS[selected_subj],savefigs=True,allinone=True,figname='Figure6_Fit_ex_estimation_'+selected_subj+'_allinone',subjin=selected_subj) # assuming that simulations are already done

            # --Plot fit to all estimation data for supplementary figures--
            if selected_subj in ESTIMATION_DATA and 'Recovery' not in ESTIMATION_DATA[selected_subj]:
                cs = {k:sims[k] for k in sims if k[0] == 'c'}
                plot_allinone_avatar(cs,simsSS,{k:ESTIMATION_DATA[k] for k in ESTIMATION_DATA if k[0] == 'c'},
                                     figname='suppl_All_subjects_avatarexercise_c',savefigs=True,nrows = len(cs)*2)
                ps={k:sims[k] for k in sims if k[0] == 'p'}
                plot_allinone_avatar(ps,
                                     simsSS,{k:ESTIMATION_DATA[k] for k in ESTIMATION_DATA if k[0] == 'p'},
                                     figname='suppl_All_subjects_avatarexercise_p',savefigs=True,nrows = len(ps)*2)
            if selected_subj in VALIDATION_DATA:
                cs = {k:sims_val[k] for k in sims_val if k[0] == 'c'}
                plot_allinone_avatar(cs,simsSS_val,{k:VALIDATION_DATA[k] for k in VALIDATION_DATA if k[0] == 'c'},
                                     figname='extra_All_subjects_avatarexercise_c_validation',savefigs=True,nrows = len(cs)*2)
                ps={k:sims_val[k] for k in sims_val if k[0] == 'p'}
                plot_allinone_avatar(ps,
                                     simsSS_val,{k:VALIDATION_DATA[k] for k in VALIDATION_DATA if k[0] == 'p'},
                                     figname='extra_All_subjects_avatarexercise_p_validation',savefigs=True,nrows = len(ps)*2)
            plt.show()


        # Create results figures for the exercise model
        if MODELNAME == 'exercise':
            # Prepare Parameters and simulations for prediction of maximum exercise and other predictions
            allparams = copy.deepcopy(best_params)
            allparams.update(best_params_val)
            all_sims = sims
            all_sims.update(sims_val)
            all_simsSS = simsSS
            all_simsSS.update(simsSS_val)

            # Create figures and a table of model simulations based on MechamP data
            if 'c1' in ESTIMATION_DATA:
                # --Model predictions for individual subjects from the MeChamP study (manuscript Figure 8)--
                # Prediction of maximum exercise inside the MRI
                subjects = None
                features = ['HR','HR change','SVi','SV change','SVR','SVR change','SBP','SBP change','DBP','DBP change','CI','CO change','ESVi','ESV change']
                alldata = ESTIMATION_DATA # add controls and patients from estimation data
                prediction_max_exercise(allparams, alldata,model,MODELNAME,features=features,
                                        figname='Figure8_Prediction_maxexercise',savefig=True,subjects=subjects,uncname="")
                
                # Prediction of same watt for all subjects (but different wmax in the different subjects)
                wmax_values=[wmax for subj, params in allparams.items() for wmax in [params[model.parameter_names.index('Wmax')]]]
                features = ['HR','HR change','SV','SV change','SVR','SVR change','SBP','SBP change','DBP','DBP change','CO','CO change','ESV','ESV change']
                activity = {'watt_in':{'values':[0,min(wmax_values),0],'time':[0,8]}} # same watt for all subjects
                prediction_max_exercise(allparams, alldata,model,MODELNAME,features=features,
                                        activity=activity,figname='Figure8_Prediction_samewatt_differentwmax',savefig=True,subjects=subjects,uncname="")

                # --Create table of feature values for mechamp data (manuscript Table 6)--
                print("\n##### Table of individual random effect changes for mechamp data #####")
                features={'keR_r':['max','keR_r','%'],'keESV_r':['max','keESV_r','%'], 'T':['max','T','s'],
                          'Workload':['max','Workload','watt'],'percWmax':['max','percWmax','%'],}
                subjects = [k for k in ESTIMATION_DATA if k[0] == 'c' or k[0] == 'p']
                if len(subjects)>0:
                    table_paramvalues(features,subjects,sims,savetables=True,tablename = 'Table6_vals_',modelname='bp_exercise_fb_both_rand')

                # ---Agreement to MeChamP estimation data for all subjects (supplementary figures)--
                selecteddata = {k:ESTIMATION_DATA[k] for k in ESTIMATION_DATA if k[0] == 'c'} 
                plot_severalfeatures(best_params,MODELNAME,model.name,sims, selecteddata, simSS=simsSS,
                                    features_to_plot=[['SBP','DBP'],['EDVi', 'ESVi','SVi'],['HR'],['SVR']],
                                    ylabels=['Blood pressure (mmHg)','LV volumes (mL/m2)','HR (bpm)','SVR (dyn.s.cm-5)'],
                                                savefig=True,figname='suppl_Fit to all controls exercise')
                selecteddata = {k:ESTIMATION_DATA[k] for k in ESTIMATION_DATA if k[0] == 'p'} 
                plot_severalfeatures(best_params,MODELNAME,model.name,sims, selecteddata, simSS=simsSS,
                                    features_to_plot=[['SBP','DBP'],['EDVi', 'ESVi','SVi'],['HR'],['SVR']],
                                    ylabels=['Blood pressure (mmHg)','LV volumes (mL/m2)','HR (bpm)','SVR (dyn.s.cm-5)'],
                                                savefig=True,figname='suppl_Fit to all patients exercise')
                
                # Plot agreement to selected estimation data (manuscript Figure 5)
                if 'average_pat' in VALIDATION_DATA:
                    selecteddata = {'average_pat':VALIDATION_DATA['average_pat'],'average_control':VALIDATION_DATA['average_control']}
                else:
                    selecteddata = {}
                selecteddata.update(lowest_highest_estimation)
                plot_severalfeatures(allparams,MODELNAME,model.name,all_sims, selecteddata, simSS=all_simsSS,
                                    features_to_plot=[['SBP','DBP'],['EDVi', 'ESVi','SVi'],['HR'],['SVR']],
                                    ylabels=['Blood pressure (mmHg)','LV volumes (mL/m2)','HR (bpm)','SVR (dyn.s.cm-5)'],
                                    savefig=True,figname='Figure5 Fit to mechamp data',wide=False,figwidth=20) 

            # Create figures of agreement to literature data
            if 'Beaudry' in ESTIMATION_DATA and "Mayr" in VALIDATION_DATA:
                # Plot agreements to literature estimation and validation data (manuscript Figure 4)
                selecteddata = {k:ESTIMATION_DATA[k] for k in ESTIMATION_DATA if k[0:7] != 'average'} 
                selecteddata.update({k:val for k,val in VALIDATION_DATA.items() if k[0] != 'c' and k[0] != 'p' and k[0:7] != 'average'})
                plot_severalfeatures(allparams,MODELNAME,model.name,all_sims, selecteddata, simSS=all_simsSS,
                                    features_to_plot=[['SBP','DBP'],['EDVi', 'ESVi'],['HR'],['SVR']],
                                    ylabels=['Blood pressure (mmHg)','LV volumes (mL/m2)','HR (bpm)','SVR (dyn.s.cm-5)'],
                                    savefig=True,figname='Figure4 Fit to bp and lv data litterature est+val',wide=False,figwidth=20)
            plt.show()
        
        print("\nFinished running the model and creating result figures and tables.")
