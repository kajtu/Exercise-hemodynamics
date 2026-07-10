import pandas as pd
from scipy import stats
import numpy as np
import os
import copy
import json
from json import JSONEncoder
from functions.simfunctions import avatar_simulation, exercise_simulation
from functions.optfunctions import f_cost_PPL
import glob
from functions.format_json import format_json_file

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Clean up any infinite values before saving
def clean_json_data(obj):
    if isinstance(obj, dict):
        return {k: clean_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_data(item) for item in obj]
    elif isinstance(obj, (float, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj

def get_tabdata(data,savetables=False,varstocompare=None,stattest='t'):
    fulldata = {exp:dat['Full'] for exp,dat in data.items()}
    datadict = {}
    subj1 = list(fulldata.keys())[0]
    tabdatanames = list(fulldata[subj1]['extra'].keys())
    
    datadict['Group'] = ['Control' if subj[0] == 'c' else  subj[0] for subj in fulldata if tabdatanames[0] in fulldata[subj]['extra'] and len(subj)>3]
    datadict['Group'] = ['HT' if val == 'p' else val for i, val in enumerate(datadict['Group'])]
    datadict['Subject'] = [subj for subj in fulldata if tabdatanames[0] in fulldata[subj]['extra']]

    # add all data from the extra dictionary in full data for each subject
    for t in tabdatanames:
        if isinstance(fulldata[subj1]['extra'][t],dict):
            datadict[t] = [fulldata[subj]['extra'][t]['mean'] for subj in fulldata if t in fulldata[subj]['extra']]
        else:
            datadict[t] = [fulldata[subj]['extra'][t] for subj in fulldata if t in fulldata[subj]['extra']]
    # add some data for rest, exercise, rampdown and recovery for each subject
    datadict['sbp_home_achieved'] = []
    datadict['dbp_home_achieved'] = []
    datadict['SBP_rest'] = [data[subj]['Rest']['SBP']['mean'][0] if data[subj]['Rest']['SBP']['mean'] is not None else None for subj in data if subj in datadict['Subject']]
    datadict['DBP_rest'] = [data[subj]['Rest']['DBP']['mean'][0] if data[subj]['Rest']['DBP']['mean'] is not None else None for subj in data if subj in datadict['Subject']]
    datadict['HR_rest'] = [data[subj]['Rest']['input']['HR']['mean'] for subj in data if subj in datadict['Subject']]
    datadict['SV_rest'] = []
    datadict['SV_exercise'] = []
    datadict['SV_recovery'] = []
    datadict['SV_rampdown'] = []
    datadict['ESV_rest'] = []
    datadict['EDV_rest'] = []
    datadict['SVR_rest'] = []
    datadict['SVR_exercise'] = []
    datadict['SVR_recovery'] = []

    datadict['watt_level1'] = []
    datadict['watt_level2'] = []
    datadict['percwmax_level1'] = []
    datadict['percwmax_level2'] = []
    datadict['time_rest'] = []
    datadict['time_level1'] = []
    datadict['time_level2'] = []
    datadict['time_recovery'] = []
    datadict['estimated_wmax'] = []
    datadict['time_total']  = [data[subj]['Full']['input']['time'][-1] for subj in data if subj in datadict['Subject']]
    for i,subj in enumerate(data):
        if subj in datadict['Subject']:
            # Stroke volume
            if not data[subj]['Full']['SV'] is None and data[subj]['Full']['SV']['time'][0] <= data[subj]['Full']['input']['activitystarttime']: #not data[subj]['Rest']['input']['EDV']['mean'] is None and not data[subj]['Rest']['input']['ESV']['mean'] is None:
                datadict['SV_rest'].append(data[subj]['Full']['SV']['mean'][0])
            else:
                datadict['SV_rest'].append(None)
            if not data[subj]['Full']['SV'] is None and data[subj]['Full']['SV']['time'][1] <= data[subj]['Full']['input']['time'][2] and data[subj]['Full']['SV']['time'][1] >= data[subj]['Full']['input']['time'][1]: 
                datadict['SV_exercise'].append(data[subj]['Full']['SV']['mean'][1])
            else:
                datadict['SV_exercise'].append(None)

            datadict['SV_rampdown'].append(None)

            if not data[subj]['Full']['SV'] is None and len(data[subj]['Full']['SV']['time'])>2 and data[subj]['Full']['SV']['time'][2] >= data[subj]['Full']['input']['time'][3]: 
                datadict['SV_recovery'].append(data[subj]['Full']['SV']['mean'][2])
            else:
                datadict['SV_recovery'].append(None)

            # EDV and ESV
            if not data[subj]['Rest']['input']['ESV'] is None: 
                datadict['ESV_rest'].append(data[subj]['Rest']['input']['ESV']['mean'])
            else:
                datadict['ESV_rest'].append(None)
            if not data[subj]['Rest']['input']['EDV'] is None: 
                datadict['EDV_rest'].append(data[subj]['Rest']['input']['EDV']['mean'])
            else:
                datadict['EDV_rest'].append(None)

            # SVR
            if 'SVR' in data[subj]['Rest']['input'] and not data[subj]['Rest']['input']['SVR'] is None:
                datadict['SVR_rest'].append(data[subj]['Rest']['input']['SVR']['mean'])
            else:
                datadict['SVR_rest'].append(None)
            if 'SVR' in data[subj]['Exercise']['input'] and not data[subj]['Exercise']['input']['SVR'] is None:
                datadict['SVR_exercise'].append(data[subj]['Exercise']['input']['SVR']['mean'])
            else:
                datadict['SVR_exercise'].append(None)
            if 'SVR' in data[subj]['Recovery']['input'] and not data[subj]['Recovery']['input']['SVR'] is None:
                datadict['SVR_recovery'].append(data[subj]['Recovery']['input']['SVR']['mean'])
            else:
                datadict['SVR_recovery'].append(None)            

            # exercise levels
            datadict['watt_level1'].append(data[subj]['Full']['input']['watt'][1])
            datadict['watt_level2'].append(data[subj]['Full']['input']['watt'][2])
            datadict['time_rest'].append(data[subj]['Full']['input']['time'][1]-data[subj]['Full']['input']['time'][0])
            datadict['time_level1'].append(data[subj]['Full']['input']['time'][2]-data[subj]['Full']['input']['time'][1])
            datadict['time_level2'].append(data[subj]['Full']['input']['time'][3]-data[subj]['Full']['input']['time'][2])
            datadict['time_recovery'].append(data[subj]['Full']['input']['time'][4]-data[subj]['Full']['input']['time'][3])
                
            # assume that % of HR reserve is the same as % of Wmax, and that HRmax = 220-age if not given as an input
            maxwatt = max(data[subj]['Full']['input']['watt'])
            maxHR = max(data[subj]['Full']['HR']['mean'])
            HRrest = data[subj]['Full']['HR']['mean'][0]
            HRmax = 220-data[subj]['Full']['input']['age']
            HRmax = HRmax*0.85 # specific for supine since it is lower!
            HRmax = max(maxHR,HRmax) # HRmax cannot be smaller than the achieved maximum HR
            percHRreserve = (maxHR-HRrest)/(HRmax-HRrest) # estimation of how close to the individual maximal effort that the performed exercise was
            datadict['percwmax_level1'].append(100*percHRreserve)
            wmax = (maxwatt/percHRreserve)
            datadict['percwmax_level2'].append(100*data[subj]['Full']['input']['watt'][2]/wmax)
            datadict['estimated_wmax'].append(wmax)

    
    datadict['SBP_exercise'] = [data[subj]['Exercise']['SBP']['mean'][0] if data[subj]['Exercise']['SBP']['mean'] is not None else None for subj in data if subj in datadict['Subject']]
    datadict['DBP_exercise'] = [data[subj]['Exercise']['DBP']['mean'][0] if data[subj]['Exercise']['DBP']['mean'] is not None else None for subj in data if subj in datadict['Subject']]
    datadict['HR_exercise']  = [data[subj]['Exercise']['input']['HR']['mean'] for subj in data if subj in datadict['Subject']]

    datadict['SBP_rampdown'] = [data[subj]['Rampdown']['SBP']['mean'][0] if data[subj]['Rampdown']['SBP']['mean'] is not None else None for subj in data if subj in datadict['Subject']]
    datadict['DBP_rampdown'] = [data[subj]['Rampdown']['DBP']['mean'][0] if data[subj]['Rampdown']['DBP']['mean'] is not None else None for subj in data if subj in datadict['Subject']]
    datadict['HR_rampdown']  = [data[subj]['Rampdown']['input']['HR']['mean'] if isinstance(data[subj]['Rampdown']['input']['HR'],dict) else data[subj]['Rampdown']['input']['HR'] for subj in data if subj in datadict['Subject']]
    
    datadict['SBP_recovery'] = [data[subj]['Recovery']['SBP']['mean'][0] if data[subj]['Recovery']['SBP']['mean'] is not None else None for subj in data if subj in datadict['Subject']]
    datadict['DBP_recovery'] = [data[subj]['Recovery']['DBP']['mean'][0] if data[subj]['Recovery']['DBP']['mean'] is not None else None for subj in data if subj in datadict['Subject']]
    datadict['HR_recovery']  = [data[subj]['Recovery']['input']['HR']['mean'] for subj in data if subj in datadict['Subject']]

    datadict['CO_rest'] = [datadict['SV_rest'][i]*datadict['HR_rest'][i]/1000 if datadict['SV_rest'][i] is not None and datadict['HR_rest'][i] is not None else None for i in range(len(datadict['Subject']))]
    datadict['CO_exercise'] = [datadict['SV_exercise'][i]*datadict['HR_exercise'][i]/1000 if datadict['SV_exercise'][i] is not None and datadict['HR_exercise'][i] is not None else None for i in range(len(datadict['Subject']))]
    datadict['CO_recovery'] = [datadict['SV_recovery'][i]*datadict['HR_recovery'][i]/1000 if datadict['SV_recovery'][i] is not None and datadict['HR_recovery'][i] is not None else None for i in range(len(datadict['Subject']))]

    tabdata = pd.DataFrame(datadict) #convert to dataframe
    tabdata.sort_values(by='Subject')
    # fix errror with datatypes
    errorvars = ('weight','length','BMI','BSA','waist_circumference','sbp_office','dbp_office',
                 'egfr','natrium','potassium','alb_krea','probnp','renin','aldosteron','hb',
                 'cholesterol','non_hdl_cholesterol','hdl','ldl','tg','glucose','hba1c')
    for var in errorvars:
        if var in tabdata.keys():
            tabdata[var] = pd.to_numeric(tabdata[var], errors='coerce')
    tabdata['sex'] = tabdata['sex']-1 #so that 0=female and 1=male

    # add units and nice names
    name_unit_dict = {}
    varstoplot = ('age','weight','length','BMI','BSA','waist_circumference',
                     'SBP_home','DBP_home','HR_home','sbp_office','dbp_office','SBP_exercise','SBP_rest','HR_exercise','HR_rest',
                     'egfr','natrium','potassium','alb_krea','probnp','renin','aldosteron','hb','hematocrit',
                     'cholesterol','non_hdl_cholesterol','glucose','hba1c','DBP_exercise','SV_exercise','SV_rest','EDV_rest',
                     'ESV_rest','EF_rest','EF_exercise','ald/renin','cfPWV','OxyPeak','PerfusionTotalPeak','SVR_rest','SVR_exercise','RBCBaseline','CO_rest','CO_exercise')
    nicenames = ('Age','Weight','Length','BMI','BSA','Waist circ.',
                     'Home SBP','Home DBP','Home HR','Office SBP','Office DBP','Exercising SBP','Resting SBP','Exercising HR','Resting HR',
                     'eGFR','Na','K','Alb.-Crea. ratio','NT-proBNP','Renin','Aldosteron','Hb','Hematocrit',
                     'Cholesterol','Non-HDL chol.','Glucose','HbA1c','Exercising DBP','Exercising SV',
                     'Resting SV','Resting EDV','Resting ESV','Resting EF','Exercising EF','Ald/renin','cfPWV','OxyP','Perfusion peak','SVR at rest','SVR at exercise','RBC baseline','CO at rest','CO at exercise')
    units = ('years','kg','cm','kg/m^2','m^2','cm',
                     'mmHg','mmHg','bpm','mmHg','mmHg','mmHg','mmHg','bpm','bpm',
                     'mL/min/1.73m^2','mmol/L','mmol/L','g/mol','ng/L','mIE/L','pmol/L','g/L','%',
                     'mmol/L','mmol/L','mmol/L','mmol/mol','mmHg','mL','mL','mL','mL','%','%', 'pmol/mIE','m/s','%','','dyn*s/cm^5','dyn*s/cm^5','','L/min','L/min')
    for idx,var in enumerate(list(tabdata.keys())):
        name_unit_dict[var] = {}
        if var in varstoplot:
            name_unit_dict[var]['Name'] = nicenames[varstoplot.index(var)]
            name_unit_dict[var]['Unit'] = units[varstoplot.index(var)]
        else:
            name_unit_dict[var]['Name'] = var
            name_unit_dict[var]['Unit'] = ''

    # %% Statistics comparing groups
    stattable = {'Variable_tabdata':(),'Variable':(),'Controls':(),
              'Patients':(),'C vs HT':()}
    pre=tabdata[tabdata['Group']=='HT']
    control=tabdata[tabdata['Group']=='Control']

    if len(control)<1 or len(pre) <1:
        return tabdata,None,name_unit_dict
    
    # add sex  0=female and 1=male
    stattable['Variable_tabdata'] += ('Sex',)
    stattable['Variable'] += ('Sex (% male)',)
    stattable['Controls'] += (f"{100*sum(control['sex'])/len(control['sex']):0.0f}%",)
    stattable['Patients'] += (f"{100*sum(pre['sex'])/len(pre['sex']):0.0f}%",)
    cnames=['c'+p[1:] for p in pre['Subject']]
    cs =[control['sex'][control['Subject']==s].to_numpy()[0] for s in cnames]
    if stattest == 't':
        stattable['C vs HT'] +=(stats.ttest_ind(cs,pre[0:len(control)]['sex'].to_numpy(),nan_policy='omit').pvalue,)
    else:
        stattable['C vs HT'] +=(stats.mannwhitneyu(control['sex'].to_numpy()-pre[0:len(control)]['sex'].to_numpy(), nan_policy='omit').pvalue,)

    # add all other variables
    varstocompare_table = ('age','sex','weight','length','waist_circumference','BMI','BSA',
                    'sbp_office','dbp_office','SBP_home','DBP_home','HR_rest',
                    'SV_rest','ESV_rest','EDV_rest',
                     'glucose','hba1c',
                     'tg','cholesterol','non_hdl_cholesterol','ldl','hdl',
                     'hb','egfr')
    if varstocompare is None:
        varstocompare=varstocompare_table
    else:
        varstocompare = set(varstocompare+varstocompare_table)

    # match controls, pre and post
    c_m_pre = []
    pre_m_c = []
    for pat in pre['Subject']:
        if 'c'+pat[1:] in list(control['Subject']):
            c_m_pre.append('c'+pat[1:])
            pre_m_c.append(pat)
    control_matched_pre = control[ control['Subject'].apply(lambda x: x in c_m_pre) ]
    pre_matched_c = pre[ pre['Subject'].apply(lambda x: x in pre_m_c) ]
    for idx,var in enumerate(varstocompare):
        # add to stat table
        stattable['Variable_tabdata'] += (var,)
        stattable['Variable'] += (f"{name_unit_dict[var]['Name']} ({name_unit_dict[var]['Unit']})",)
        stattable['Controls'] += (f"{control[var].mean():0.1f}+-{control[var].std():0.1f}",)
        stattable['HT'] +=  (f"{pre[var].mean():0.1f}+-{pre[var].std():0.1f}",)
        diff = control_matched_pre[var].to_numpy()-pre_matched_c[var].to_numpy()
        if stattest == 't':
            stattable['C vs HT'] += (stats.ttest_ind(control_matched_pre[var].to_numpy(), pre_matched_c[var].to_numpy(), nan_policy='omit').pvalue,)
        else:
            stattable['C vs HT'] += (stats.mannwhitneyu(diff[~np.isnan(diff)], y=None).pvalue,)
    stattable = pd.DataFrame(stattable)
    print(stattable)

    if savetables:
        tabdata.to_excel("./data/characteristics_individuals.xlsx",sheet_name='redcapdata') 
        stattable.to_excel("./data/table1_characteristics.xlsx",sheet_name='redcapdata')
    return tabdata,stattable,name_unit_dict

def colorpicker(patnum,backupcolor="#000000ff",model=None, rgb=False):
    if model is None or model== 'exercise' or model == 'avatar' or model == 'avatar_exercise':
        if patnum == 'HT' or (patnum[0] == 'p' and len(patnum)<4):
            col = '#d28383ff' 
        elif patnum == 'Control' or patnum[0] == 'c':
            col = '#888da8ff' 
        elif patnum == 'Post' or (patnum[0] == 'p' and len(patnum)>4):
            col = "#f4a582"
        elif patnum == 'Rest':
            col = '#d81b60ff'
        elif patnum == 'Exercise':
            col = '#ff8f00ff'
        elif patnum == 'Rampdown':
            col = '#856d7aff'
        elif patnum == 'Recovery':
            col = '#614ca9ff'
        elif patnum == 'RecoveryEnd':
            col = '#616786ff'
        elif patnum == 'average_control':
            col = "#4255c5ff"
        elif patnum == 'average_pat' or patnum == 'average_pat_pre':
            col = "#d23b3bff"
        elif patnum == 'average_pat_post':
            col = "#f4a582"
        elif patnum == 'Mayr' or patnum == 'Roberts':
            col = "#0ba185ff"
        else:
            col = backupcolor
    else:
        col = backupcolor

    if rgb:
        return hex_to_rgb(col)
    else:
        return col


def get_BSA(inputdata):
    if 'BSA' in inputdata and inputdata['BSA'] is not None:
        BSA = inputdata['BSA']
    elif 'weight' in inputdata and inputdata['weight'] is not None and 'height' in inputdata and inputdata['height'] is not None:
        BSA=np.sqrt( inputdata['weight']* inputdata['height']*100)/60
    elif 'weight' in inputdata and inputdata['weight'] is not None:
        BSA=np.sqrt( inputdata['weight']* 175)/60  # default height: 175 cm
    elif 'height' in inputdata and inputdata['height'] is not None:
        BSA=np.sqrt(80* inputdata['height']*100)/60 # default weight: 80 kg
    else:
        BSA = np.sqrt(80*175)/60 #default value if no wieht or height info is given
    return BSA


def hex_to_rgb(hex_color):
    return np.array([int(hex_color[i:i+2], 16) for i in (1, 3, 5)]) / 255


def setup_PPL(model, modelname,sims, simsSS, data, chi2_limit, validation_experiments,default_params,common_params,pred_features=[]):
    from functions.simfunctions import load_best_parameters_all
    # Initialize folder
    if modelname == 'avatar_exercise':
        subj = list(data.keys())[0]
        data = data[subj]
        os.makedirs(f"./results_PPL/avatar_exercise/{subj}", exist_ok=True)
    else:
        os.makedirs(f"./results_PPL/{model.name}", exist_ok=True)
    # max_tfs to consider for PPL calculation for each feature 
    max_tfs = 10
    
    # do not calculate cost for validation experiments - remove them from data
    ALL_DATA_PPL = copy.deepcopy(data)
    for k_exp in validation_experiments:
        ALL_DATA_PPL.pop(k_exp)
    if modelname == 'avatar_exercise':
        ALL_DATA_PPL = {subj:ALL_DATA_PPL}

    onlyonetf = ['Caa','ELCo','Emax_LV','Emax_LA','m2_LV','k_syst_LV',
                         'onset_LA','k_diast_LV','Rleg','Rupper','Rlower',
                         'k_diast_LV_change','k_syst_LV_change','m2_LV_change',
                         'Caa_change','onset_LA_change','Rleg_change','Rlower_change',
                         'Rul_change','Rlu_change','Emax_LV_change','Emax_LA_change',
                         'Rupper_change','Rtot_change','Rbody_change',
                         'Imax_HR_r', 'Imax_EDV_r', 'Imax_ESV_r', 'Imax_R_r',
                         'Imax_HR_rval', 'Imax_EDV_rval', 'Imax_ESV_rval', 'Imax_R_rval']

    # find number of combinations to iterate over
    PPL_combinations = {}
    iter = 0
    for k_exp, d in data.items():
        if isinstance(d,dict) and 'input' not in d:
            print(f'setupPPL: {k_exp} contains several experiment, choosing the first to calculate PPL combinations')
            d = list(d.values())[0]
        all_featurenames = []
        # no reasons to look for the PPL value if no activity has started yet - i.e. we are in steady state
        if modelname == "exercise":
            t_first_activity = d['input']['activitystarttime']
        else:
            t_first_activity=-1
        for k_obs, obs in d.items():
            if k_obs not in ["input","extra","meta"] and obs['mean'] is not None and k_obs+'i' not in all_featurenames and k_obs[:-1] not in all_featurenames:
                obstime = copy.deepcopy(obs['time'])
                if k_obs in onlyonetf:
                    obstime = [obstime[-1]] # only take the last time point for features that only have one value
                elif len(obstime)>max_tfs:
                    obstime = obstime[::max(1,int(len(obstime)/max_tfs))] # if more than max_tfs time points, only take a subset to reduce computation time
                for t in obstime:
                    if t >= t_first_activity:
                        PPL_combinations[iter] = {"k_exp": k_exp, "k_obs": k_obs, "time": t}
                        iter += 1
                        if k_obs not in all_featurenames:
                            all_featurenames.append(k_obs)
                if t < t_first_activity and k_obs in pred_features  and k_obs in model.feature_names and k_obs+'i' not in all_featurenames and k_obs[:-1] not in all_featurenames:
                    # if the feature is in pred_features, we still want to add it to the combinations
                    # even if the first data point is before the first activity
                    obstime = copy.deepcopy(d['input']['simulationtime'])
                    if len(obstime)>max_tfs:
                        obstime = obstime[::max(1,int(len(obstime)/max_tfs))] # if more than 30 time points, only take a subset to reduce computation time
                    for t in obstime:
                        if t >= t_first_activity:
                            PPL_combinations[iter] = {"k_exp": k_exp, "k_obs": k_obs, "time": t}
                            iter += 1
                            if k_obs not in all_featurenames:
                                all_featurenames.append(k_obs)
        # also add combinations for the wanted predicted features
        for feat in pred_features:
            if feat not in all_featurenames and feat in model.feature_names and feat+'i' not in all_featurenames and feat[:-1] not in all_featurenames:
                if modelname == 'avatar_exercise' and feat[-6:] == 'change' and k_exp=='Rest':
                    continue # do not calculate change in rest, only in exercise
                obstime = copy.deepcopy(d['input']['simulationtime'])
                if feat in onlyonetf:
                    obstime = [obstime[-1]] # only take the last time point for features that only have one value
                if len(obstime)>max_tfs:
                    obstime = obstime[::max(1,int(len(obstime)/max_tfs))] # if more than max_tfs time points, only take a subset to reduce computation time
                for t in obstime:
                    if t >= t_first_activity:
                        PPL_combinations[iter] = {"k_exp": k_exp, "k_obs": feat, "time": t}
                        iter += 1
                        if feat not in all_featurenames:
                            all_featurenames.append(feat)

    # load best solution
    subjects = list(data.keys())
    # The best parameters found so far with respect to the agreement to data:
    if modelname == 'avatar_exercise':
        theta_best = load_best_parameters_all([subj],f"./parameters/avatar_exercise", cost_key='f', model=model,default_params=default_params,common_params=common_params,paramnames=model.parameter_names)
        modelfolder = "avatar_exercise/"+subj
    else:
        theta_best = load_best_parameters_all(subjects,f"./parameters/{model.name}", cost_key='f', model=model,default_params=default_params,common_params=common_params,paramnames=model.parameter_names)
        modelfolder = model.name
    # PPL objective function to find PPL value for the best parameters
    # cannot be imported due to circular dependencies
    from functions.simfunctions import f_cost_all, f_cost, estimate_sbp
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
                idx = sim.feature_names.index(obs_to_optimize)
                tidx =min(np.searchsorted(sim.time_vector, t_to_optimize),len(sim.time_vector)-1)
                PPL_value = sim.feature_data[tidx, idx]
        except Exception as e:
            # Handle any errors and return a large PPL value
            print(f"Error in {exp_to_optimize} {obs_to_optimize} {t_to_optimize}: {e} (fcostppl setup)")
            PPL_value = 1e20
                        
        return PPL_value, cost_this_exp, cost_all_exps
        
    PPL_combinations_all = copy.deepcopy(PPL_combinations)
    # Iterate over all combinations of experiments, observations and time points
    # and calculate the PPL value for the best parameters 
    for iter in PPL_combinations_all:
        k_obs = PPL_combinations_all[iter]['k_obs']
        k_exp = PPL_combinations_all[iter]['k_exp']
        t = PPL_combinations_all[iter]['time']
        
        # Initialize subfolder
        if not os.path.exists(f"./results_PPL/{modelfolder}/{k_exp}_{k_obs}_{t}"):
            os.makedirs(f"./results_PPL/{modelfolder}/{k_exp}_{k_obs}_{t}",exist_ok=True)
                    
            # find PPL cost and save the PPL objective value if the cost is lower than chi2-limit
            PPL_obj, cost_this_exp, cost_all_exps = f_cost_PPL(theta_best, sims, ALL_DATA_PPL, modelname, k_exp, k_obs, t,simsSS=simsSS,print_costs=False)
            if cost_all_exps > chi2_limit:
                print(f"setupPPL: cost {cost_all_exps} > limit {chi2_limit}: not saving value for: {k_exp}-{k_obs}-time_value-{t}")
            else:
                try:
                    with open(f"./results_PPL/{modelfolder}/{k_exp}_{k_obs}_{t}/{k_exp}_{k_obs}_{t} initial({PPL_obj:e}).json", 'w') as file:
                        out = {"f": PPL_obj, "x": theta_best}
                        json.dump(out, file, cls=NumpyArrayEncoder)
                except PermissionError:
                    print(f"Permission denied for file './results_PPL/{modelfolder}/{k_exp}_{k_obs}_{t}/{k_exp}_{k_obs}_{t}.json'. Not saving current solution.")

            # also save a file to keep track of if the time point has been minimized/maximized already
            bool_dict = {"minimized": False, "maximized": False}
            with open(f"./results_PPL/{modelfolder}/{k_exp}_{k_obs}_{t}/min_max.json", 'w') as file:
                json.dump(bool_dict, file, cls=NumpyArrayEncoder)
        elif not os.path.exists(f"./results_PPL/{modelfolder}/{k_exp}_{k_obs}_{t}/min_max.json"):
            bool_dict = {"minimized": False, "maximized": False}
            with open(f"./results_PPL/{modelfolder}/{k_exp}_{k_obs}_{t}/min_max.json", 'w') as file:
                json.dump(bool_dict, file, cls=NumpyArrayEncoder)
        else:
            try:
                # load file that checks progress
                with open(f"./results_PPL/{modelfolder}/{k_exp}_{k_obs}_{t}/min_max.json") as ff:
                    min_max = json.load(ff)
                # check if PPL value has already been estimated - if yes pop the index
                if min_max["minimized"] == True and min_max["maximized"] == True:
                    PPL_combinations.pop(iter)
            except Exception as e:
                print(f"Could not load min_max.json file in {modelfolder}/{k_exp}_{k_obs}_{t}, error: {e}.")
    del PPL_combinations_all
    return ALL_DATA_PPL, PPL_combinations

def get_UC(UC_data, obs, extra_exp, sims, modelname, model,data):
    from functions.simfunctions import estimate_sbp
    # calculate uncertainty depending on the observation obs
    # some require a bit of calculation (like maximum value etc)
    if obs in ['PV_LV','PV_LA']:
        # find maximum and minimum distance from the middle point of the PV loop for each time point
        if obs == 'PV_LV':
            xvals = sims[extra_exp].feature_data[:, sims[extra_exp].feature_names.index('Vlv')]
            yvals = sims[extra_exp].feature_data[:, sims[extra_exp].feature_names.index('P_LV')]
        elif obs == 'PV_LA':
            xvals = sims[extra_exp].feature_data[:, sims[extra_exp].feature_names.index('Vla')]
            yvals = sims[extra_exp].feature_data[:, sims[extra_exp].feature_names.index('P_LA')]
        distance = np.sqrt((xvals-UC_data[extra_exp][obs]["midpoint"][0])**2 + (yvals-UC_data[extra_exp][obs]["midpoint"][1])**2)
        indsmin = np.where(distance < UC_data[extra_exp][obs]["Min_dist"])[0]
        indsmax = np.where(distance > UC_data[extra_exp][obs]["Max_dist"])[0]
        # save coordinates for min and max distance
        UC_data[extra_exp][obs]["Max_dist"] = np.maximum(UC_data[extra_exp][obs]["Max_dist"], distance)
        UC_data[extra_exp][obs]["Min_dist"] = np.minimum(UC_data[extra_exp][obs]["Min_dist"], distance)
        UC_data[extra_exp][obs]["Max_y"][indsmax] =  yvals[indsmax]
        UC_data[extra_exp][obs]["Max_x"][indsmax] =  xvals[indsmax]
        UC_data[extra_exp][obs]["Min_y"][indsmin] =  yvals[indsmin]
        UC_data[extra_exp][obs]["Min_x"][indsmin] =  xvals[indsmin]
                            
    #  special case for SBP and DBP etc for avatar model
    elif (modelname == 'avatar' or modelname == 'avatar_exercise') and obs in ['Emax_LV','Emax_LA','SBP','DBP','LAP','LVP','ESV','EDV']:
        featnames = ['Emax_LV','Emax_LA','SBP','DBP','LAP','LVP','ESV','EDV']
        varnames =  ['Elv','Ela','Pao','Pao','P_LA','P_LV','Vlv','Vlv']
        ysim = sims[extra_exp].feature_data[:, sims[extra_exp].feature_names.index(varnames[featnames.index(obs)])]
        if obs in ['Emax_LV','Emax_LA','EDV']:
            y_sim = max(ysim)
        elif obs in ['ESV','DBP']:
            y_sim = min(ysim)
        elif obs in ['LAP','LVP']:
              y_sim = np.mean(ysim)
        elif obs in ['SBP']:
            if 'SBP' in data[extra_exp]:
                sbp_data = data[extra_exp]['SBP']['mean']
            else:
                sbp_data = None
            y_sim = estimate_sbp(sims[extra_exp],sbp_data)
        # save the uncertainty data (min and max of all simulations for that observable)
        UC_data[extra_exp][obs]["Max"] = np.maximum(UC_data[extra_exp][obs]["Max"], y_sim)
        UC_data[extra_exp][obs]["Min"] = np.minimum(UC_data[extra_exp][obs]["Min"], y_sim)
    # change from rest to exercise for some parameters in avatar_exercise
    elif extra_exp == 'Exercise' and modelname == 'avatar_exercise' and obs in ['Emax_LV_change','Emax_LA_change','Rupper_change','Rtot_change','Rbody_change']:
        if obs == 'Emax_LV_change':
            idx = sims[extra_exp].feature_names.index('Elv')
        elif obs == 'Emax_LA_change':
            idx = sims[extra_exp].feature_names.index('Ela')
        elif obs == 'Rupper_change':
            idx = sims[extra_exp].feature_names.index('Rupper')
        elif obs == 'Rtot_change':
            idx = sims[extra_exp].feature_names.index('Rtot')
        elif obs == 'Rbody_change':
                idx = sims[extra_exp].feature_names.index('Rbody')
        rval = max(sims['Rest'].feature_data[:, idx])
        eval = max(sims['Exercise'].feature_data[:, idx])
        y_sim = (eval - rval)/rval*100
        # save the uncertainty data (min and max of all simulations for that observable)
        UC_data[extra_exp][obs]["Max"] = np.maximum(UC_data[extra_exp][obs]["Max"], y_sim)
        UC_data[extra_exp][obs]["Min"] = np.minimum(UC_data[extra_exp][obs]["Min"], y_sim)
    elif obs in sims[extra_exp].feature_names and obs in UC_data[extra_exp]:
        y_sim = sims[extra_exp].feature_data[:, model.feature_names.index(obs)] 
        # save the uncertainty data (min and max of all simulations for that observable)
        UC_data[extra_exp][obs]["Max"] = np.maximum(UC_data[extra_exp][obs]["Max"], y_sim)
        UC_data[extra_exp][obs]["Min"] = np.minimum(UC_data[extra_exp][obs]["Min"], y_sim)

def create_UCstructure(data,UC_data,all_observables, model, modelname, sims, simsSS, default_params, pred_features=[], t_step=0.1,subject=None):
    onlyonetf = ['Caa','ELCo','Emax_LV','Emax_LA','m2_LV','k_syst_LV',
                         'onset_LA','k_diast_LV','Rleg','Rupper','Rlower',
                         'k_diast_LV_change','k_syst_LV_change','m2_LV_change',
                         'Caa_change','onset_LA_change','Rleg_change','Rlower_change',
                         'Rul_change','Rlu_change','Emax_LV_change','Emax_LA_change',
                         'Rupper_change','Rtot_change','Rbody_change',
                         'SBP','DBP','ESV','EDV']
    for k_exp, d in data.items():
        UC_data[k_exp] = {}

        if isinstance(d,dict) and 'input' not in d:
            print(f'collect_PPL_uncertainty: {k_exp} contains several experiment, adding all of them to create UC structure')
            create_UCstructure(d,UC_data[k_exp],all_observables, model, modelname, sims, simsSS, default_params, pred_features=pred_features, t_step=t_step)
            continue

        t_UC = np.arange(d['input']['simulationtime'][0],d['input']['simulationtime'][-1]+t_step,t_step) # sim highres for plotting
        # add all features in data to the UC structure
        for k_obs, _ in d.items():
            if k_obs in ["input", "meta", "extra"]:
                continue
            UC_data[k_exp][k_obs] = {}            
            UC_data[k_exp][k_obs]["Time"] = t_UC
            UC_data[k_exp][k_obs]["Max"] = [float('-inf')] * len(t_UC)
            UC_data[k_exp][k_obs]["Min"] = [float('inf')]  * len(t_UC)

            if k_obs[:-1] in model.feature_names:
                # if the observable is indexed, add the non-indexed observable as well
                UC_data[k_exp][k_obs[:-1]] = copy.deepcopy(UC_data[k_exp][k_obs])
                if k_obs[:-1] not in all_observables:
                    all_observables.append(k_obs[:-1])
            elif k_obs+"i" in model.feature_names:
                # if the observable is not indexed, add the indexed observable as well
                UC_data[k_exp][k_obs+"i"] = copy.deepcopy(UC_data[k_exp][k_obs])
                if k_obs+"i" not in all_observables:
                    all_observables.append(k_obs+"i")

            if k_obs not in all_observables:
                all_observables.append(k_obs)

        # add all features in pred_features to the UC structure as well
        UC_data[k_exp]["Time"] = t_UC
        for feat in pred_features:
            if modelname == 'avatar_exercise' and feat[-6:] == 'change' and k_exp=='Rest':
                    continue # do not calculate change in rest, only in exercise
            if feat not in d.keys() and (feat in model.feature_names or feat in ['PV_LV','PV_LA']):
                UC_data[k_exp][feat] = {}
                if feat in ['PV_LV','PV_LA']:
                    UC_data[k_exp][feat]["Max_x"] = np.array([float('-inf')] * len(t_UC))
                    UC_data[k_exp][feat]["Min_x"] = np.array([float('inf')]  * len(t_UC))
                    UC_data[k_exp][feat]["Max_y"] = np.array([float('-inf')] * len(t_UC))
                    UC_data[k_exp][feat]["Min_y"] = np.array([float('inf')]  * len(t_UC))
                    UC_data[k_exp][feat]["Min_dist"] = np.array([float('inf')]  * len(t_UC))
                    UC_data[k_exp][feat]["Max_dist"] = np.array([float('-inf')]  * len(t_UC))
                    # mid point of the PV loop (x,y), needed to calculate distance
                    if modelname == 'avatar_exercise':
                        avatar_simulation(default_params[subject], sims[k_exp],simsSS[k_exp],timepoints=t_UC)
                    else:
                        avatar_simulation(default_params[k_exp], sims[k_exp],simsSS[k_exp],timepoints=t_UC)
                    if feat == 'PV_LV':
                        x = sims[k_exp].feature_data[:, sims[k_exp].feature_names.index('Vlv')]
                        y = sims[k_exp].feature_data[:, sims[k_exp].feature_names.index('P_LV')]
                    elif feat == 'PV_LA':
                        x = sims[k_exp].feature_data[:, sims[k_exp].feature_names.index('Vla')]
                        y = sims[k_exp].feature_data[:, sims[k_exp].feature_names.index('P_LA')]
                    UC_data[k_exp][feat]["midpoint"] = [np.mean(x),np.mean(y)]
                elif (modelname == 'avatar' or modelname == 'avatar_exercise') and feat in onlyonetf:
                    UC_data[k_exp][feat]["Max"] = float('-inf')
                    UC_data[k_exp][feat]["Min"] = float('inf')
                else:
                    UC_data[k_exp][feat]["Max"] = [float('-inf')] * len(t_UC)
                    UC_data[k_exp][feat]["Min"] = [float('inf')]  * len(t_UC)
                    
                if feat not in all_observables:
                    all_observables.append(feat)

                if feat[:-1] in model.feature_names and feat[:-1] not in UC_data[k_exp]:
                    # if the observable is indexed, add the non-indexed observable as well
                    UC_data[k_exp][feat[:-1]] = copy.deepcopy(UC_data[k_exp][feat])
                    if feat[:-1] not in all_observables:
                        all_observables.append(feat[:-1])
                elif feat+"i" in model.feature_names  and feat+"i" not in UC_data[k_exp]:
                    # if the observable is not indexed, add the indexed observable as well
                    UC_data[k_exp][feat+"i"] = copy.deepcopy(UC_data[k_exp][feat])
                    if feat+"i" not in all_observables:
                        all_observables.append(feat+"i")

                # if the "change" observable is present, also save the min and max for the change observable
                if modelname == "exercise":
                    if feat + " change" in model.feature_names and feat + " change" not in UC_data[k_exp]:
                        UC_data[k_exp][feat + " change"] = copy.deepcopy(UC_data[k_exp][feat])
                        if feat + " change" not in all_observables:
                            all_observables.append(feat + " change")
                    elif feat.replace(" change", "") in model.feature_names and feat.replace(" change", "") not in UC_data[k_exp]:
                        UC_data[k_exp][feat.replace(" change", "")] = copy.deepcopy(UC_data[k_exp][feat])
                        if feat.replace(" change", "") not in all_observables:
                            all_observables.append(feat.replace(" change", ""))
                    # if the "decrease" observable is present, also save the min and max for the decrease observable
                    if feat + " decrease" in model.feature_names and feat + " decrease" not in UC_data[k_exp]:
                        UC_data[k_exp][feat + " decrease"] = copy.deepcopy(UC_data[k_exp][feat])
                        if feat + " decrease" not in all_observables:
                            all_observables.append(feat + " decrease")
                    elif feat.replace(" decrease", "") in model.feature_names and feat.replace(" decrease", "") not in UC_data[k_exp]:
                        UC_data[k_exp][feat.replace(" decrease", "")] = copy.deepcopy(UC_data[k_exp][feat])
                        if feat.replace(" decrease", "") not in all_observables:
                            all_observables.append(feat.replace(" decrease", ""))

    
def collect_PPL_uncertainty(default_params,model, modelname, data, sims,simscost, simsSS,chi2lims,
                            validation_experiments=[],pred_features=[],COMMON_PARAMETERS=[],
                            all_data=None,SUBJECT_SPECIFIC=False, pred_sims = None):
    from functions.simfunctions import estimate_sbp, bp_punishment_exercise
    if modelname == "exercise":
        t_step = 0.1
        modelfoldername = model.name
    elif modelname == "avatar":
        t_step = 0.001
        modelfoldername = model.name
    elif modelname == "avatar_exercise":
        #run one subject at a time, flag subjectspecific=False
        t_step = 0.001
        subject = list(data.keys())[0]
        data = data[subject]
        modelfoldername = "avatar_exercise"+"/"+subject
        sims = sims[subject]
    else:
        raise ValueError(f"Model name {modelname} is not supported for uncertainty collection.")
    
    if modelname == "avatar_exercise":
        simsSScost = simsSS
        simsSS = simsSS[subject] 
    else:
        simsSScost = simsSS
        subject=None
    
    # create a dictionary to store results
    # alldata does not contain validation experiments
    if all_data is None:
        all_data = data

    UC_data = {}
    all_observables = []
    create_UCstructure(data,UC_data,all_observables, model, modelname, sims, simsSS, default_params, pred_features=pred_features, t_step=t_step,subject=subject)

    # If there is any predictions, create a separate dicitonary to store the prediciton results
    UC_pred_data = {}
    if pred_sims is not None:
        for predname, predsims in pred_sims.items():
            UC_pred_data[predname] = {}
            for exp in predsims:
                if predname[0:12] == 'contribution':
                    UC_pred_data[predname][exp] = {}
                    observables=['full','param_change','contribution']
                    for obs in observables:
                        UC_pred_data[predname][exp][obs] = {}
                        UC_pred_data[predname][exp][obs]["Max"] = float('-inf')
                        UC_pred_data[predname][exp][obs]["Min"] = float('inf')
                else:
                    UC_pred_data[predname][exp] = copy.deepcopy(UC_data[exp])
                    UC_pred_data[predname][exp]["Time"] = pred_sims[predname][exp].time_vector
                    for obs in UC_pred_data[predname][exp]:
                        if obs not in ["Time"]:
                            UC_pred_data[predname][exp][obs]["Max"] = [float('-inf')] * len(UC_pred_data[predname][exp]["Time"])
                            UC_pred_data[predname][exp][obs]["Min"] = [float('inf')] * len(UC_pred_data[predname][exp]["Time"])

    # loop over the PPL results folder and add the results to the UC structure
    root_directory = f"./results_PPL/{modelfoldername}"
    subdirs = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    if modelname == 'avatar_exercise':
        os.makedirs(f"parameters/avatar_exercise/UC", exist_ok=True)
    else:
        os.makedirs(f"parameters/{modelfoldername}/UC", exist_ok=True)

    nfiles=0
    sdirlast=''
    # iterate over every subdirectory
    for sind, subdir in enumerate(subdirs):
        # Find time point from the folder name
        time_point = float(subdir.split("_")[-1])

        # Find which experiment and observable this subdir belongs to
        id_obs = [subdir.split("_").index(x) for x in all_observables if x in subdir.split("_")]
        if len(id_obs)>0:
            id_obs = id_obs[0]
            k_obs = "_".join(subdir.split("_")[id_obs:-1])
            k_exp = "_".join(subdir.split("_")[:id_obs])
        elif subdir.split("_")[1] + "_" + subdir.split("_")[2] in all_observables:
            k_obs = subdir.split("_")[1] + "_" + subdir.split("_")[2]
            k_exp = subdir.split("_")[0]
        else:
            print(f"Could not identify observable in {subdir}, skipping this folder.")
            continue

        # skip subdir if it is not in data or observable not in data or observable not in model or experiment not in sims or simscost
        if k_exp not in UC_data or k_obs not in UC_data[k_exp] or k_obs not in model.feature_names:
            continue
        if modelname != 'avatar_exercise' and (k_exp not in sims or k_exp not in simscost):
            continue
        print(f"Processing subdir {sind}/{len(subdirs)}: {subdir}")
        # iterate over every file
        files = glob.glob(os.path.join(root_directory, subdir, "*.json"))
        for file in files:
            if file.endswith(".json") and 'min_max' not in file and 'initial' not in file:
                try:  
                    with open(f"{file}", 'r') as f:
                        res = json.load(f)
                        
                    # get parameter vector (dict of parameters for all experiments/subjects)
                    try:
                        p = res['x']
                    except:
                        print(f"Could not read parameter vector from {file}, skipping this file.")
                        continue
                    # make sure that cost is under chi2 limit
                    if SUBJECT_SPECIFIC:
                        value_accepted, cost_accepted_this,cost_accepted_all = f_cost_PPL(p, {k_exp:simscost[k_exp]}, {k_exp:all_data[k_exp]}, modelname,k_exp, k_obs, time_point,simsSS={k_exp:simsSScost[k_exp]},print_costs=False)
                    else:
                        value_accepted, cost_accepted_this,cost_accepted_all = f_cost_PPL(p, simscost, all_data, modelname,k_exp, k_obs, time_point,simsSS=simsSScost,print_costs=False)
                    if np.round(cost_accepted_this,2) > np.round(chi2lims[k_exp],2) or np.round(cost_accepted_all,2) > np.round(chi2lims['all'],2):
                        # remove the file if the cost is above the chi2 limit
                        print(f"File {file} has cost {cost_accepted_this}>{chi2lims[k_exp]}  and/or {cost_accepted_all}>{chi2lims['all']} for experiment {k_exp}. Deleting this file.")
                        os.remove(file)
                        continue # skip this file if the cost is above the chi2 limit
                    
                    print(f'Simulating file {nfiles}: {file}')
                    nfiles+=1
                    # if the parameter set passes chi2, simulate all experiments with these parameters
                    for extra_exp in UC_data:
                        if modelname == 'avatar_exercise':
                            params_to_sim = p[subject]
                        elif extra_exp == k_exp:
                            params_to_sim = p[k_exp]
                        elif extra_exp in p and extra_exp in sims and extra_exp not in ['Ikeda_50_10','Ikeda_50']:
                            params_to_sim = p[extra_exp]
                            # since default ic changed after saving the files, ikeda validation needs to be updated
                        elif extra_exp in sims and extra_exp in default_params:
                            # add all common parameters to the parameter vector for the extra experiment
                            params_to_sim = copy.deepcopy(default_params[extra_exp])
                            for param in COMMON_PARAMETERS:
                                params_to_sim[model.parameter_names.index(param)] = copy.deepcopy(p[k_exp][model.parameter_names.index(param)])
                            p[extra_exp] = params_to_sim
                        # check if the cost is ok also for the extra experiment if not a validation exp
                        # (only used for the random effects parameters in the exercise model)
                        if extra_exp != k_exp and extra_exp not in validation_experiments:
                            if SUBJECT_SPECIFIC:
                                value_accepted, cost_accepted_this,cost_accepted_all = f_cost_PPL(p, {extra_exp:simscost[extra_exp]}, {extra_exp:all_data[extra_exp]}, modelname,extra_exp, k_obs, time_point,simsSS={extra_exp:simsSScost[extra_exp]},print_costs=False)
                            else:
                                value_accepted, cost_accepted_this,cost_accepted_all = f_cost_PPL(p, simscost, all_data, modelname,extra_exp, k_obs, time_point,simsSS=simsSScost,print_costs=False)
                            if np.round(cost_accepted_this,2) > np.round(chi2lims[extra_exp],2) or np.round(cost_accepted_all,2) > np.round(chi2lims['all'],2):
                                continue # skip the parameters for the extra experiment if the cost is above the chi2 limit
                                    
                        # simulate the model for the parameters from the current file
                        if not isinstance(sims[extra_exp],dict):
                            t_high_res = UC_data[extra_exp]["Time"]
                        try:
                            if modelname == 'exercise':
                                exercise_simulation(params_to_sim, sims[extra_exp],simsSS[extra_exp],timepoints=t_high_res)
                            elif modelname == 'avatar' or modelname == 'avatar_exercise':
                                avatar_simulation(params_to_sim, sims[extra_exp],simsSS[extra_exp],timepoints=t_high_res)
                        except Exception as e:
                            print(f"Error in simulating params from {file}: {e}")
                            continue
                        
                        if UC_pred_data:
                            to_test_contributions = {'HR':['Imax_R','Imax_R_r','Imax_ESV','Imax_ESV_r','Imax_EDV','Imax_EDV_r'],
                                                'SVR':['Imax_HR','Imax_HR_r','Imax_ESV','Imax_ESV_r','Imax_EDV','Imax_EDV_r'],
                                                'SV':['Imax_HR','Imax_HR_r','Imax_R','Imax_R_r'],
                                                'EDV':['Imax_HR','Imax_HR_r','Imax_R','Imax_R_r','Imax_ESV','Imax_ESV_r'],
                                                'ESV':['Imax_HR','Imax_HR_r','Imax_R','Imax_R_r','Imax_EDV','Imax_EDV_r']}
                            for pred in pred_sims:
                                # each prediction
                                if extra_exp in pred_sims[pred]:
                                    # each experiment
                                    try:
                                        if modelname == 'exercise':
                                            exercise_simulation(params_to_sim, pred_sims[pred][extra_exp],simSS=None)
                                            bp_error = bp_punishment_exercise(pred_sims[pred][extra_exp])
                                            if bp_error > 0:
                                                continue # skip this prediction if there is a bp error (unrealistic bp response)
                                        elif modelname == 'avatar' or modelname == 'avatar_exercise':
                                            avatar_simulation(params_to_sim, pred_sims[pred][extra_exp],simSS=None)
                                    except Exception as e:
                                        print(f"Error in simulating prediction {pred} for exp {extra_exp} with params from {file}: {e}")
                                        continue


                        # extract min/max of all observables at all time points from the simulated parameters 
                        for obs in all_observables:
                            if isinstance(sims[extra_exp],dict):
                                for simname in sims[extra_exp]:
                                    if obs in UC_data[extra_exp][simname] and k_obs in UC_data[extra_exp][simname]:
                                        get_UC(UC_data[extra_exp], obs, simname, sims[extra_exp], modelname, model,data)
                            elif obs not in UC_data[extra_exp] or k_obs not in UC_data[extra_exp]:
                                continue
                            else:
                                get_UC(UC_data, obs, extra_exp, sims, modelname, model,data)
                        
                                # unceratainty for predictions only for the loaded observable k_obs
                                if UC_pred_data:
                                    for pred in pred_sims:
                                        if extra_exp in pred_sims[pred] and obs in UC_pred_data[pred][extra_exp]:
                                            get_UC(UC_pred_data[pred], obs, extra_exp, pred_sims[pred], modelname, model,data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {file}")

        #save the uncertainty after doing all simulations within every 100th folder
        if np.mod(nfiles+1,100)==0 and sdirlast != subdir:
            sdirlast=subdir
            print(f"Saving intermediate uncertainty results after processing {sind+1} of {len(subdirs)} folders and {nfiles} files.")
            if modelname == "avatar_exercise":
                with open(f"parameters/avatar_exercise/UC/UC_PPL_{subject}.json", 'w') as f:
                    json.dump(UC_data, f, cls=NumpyArrayEncoder)
                format_json_file(f"parameters/avatar_exercise/UC/UC_PPL_{subject}.json", f"parameters/avatar_exercise/UC/UC_PPL_{subject}.json")
            else:
                with open(f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}.json", 'w') as f:
                    json.dump(UC_data, f, cls=NumpyArrayEncoder)
                format_json_file(f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}.json", f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}.json")

            if UC_pred_data:
                with open(f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}_pred.json", 'w') as f:
                    json.dump(UC_pred_data, f, cls=NumpyArrayEncoder)
                format_json_file(f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}_pred.json", f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}_pred.json")

    # save the final uncertainty results
    print('Saving final uncertainty results...')
    if modelname == "avatar_exercise":
        with open(f"parameters/avatar_exercise/UC/UC_PPL_{subject}.json", 'w') as f:
            json.dump(clean_json_data(UC_data), f, cls=NumpyArrayEncoder)
        format_json_file(f"parameters/avatar_exercise/UC/UC_PPL_{subject}.json", f"parameters/avatar_exercise/UC/UC_PPL_{subject}.json")
    else:
        with open(f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}.json", 'w') as f:
            json.dump(clean_json_data(UC_data), f, cls=NumpyArrayEncoder)
        format_json_file(f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}.json", f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}.json")

    if UC_pred_data:
        with open(f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}_pred.json", 'w') as f:
            json.dump(clean_json_data(UC_pred_data), f, cls=NumpyArrayEncoder)
        format_json_file(f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}_pred.json", f"parameters/{modelfoldername}/UC/UC_PPL_{modelfoldername}_pred.json")


def create_exercise_prediction(activity,data,model):
    import sund
    from functions.simfunctions import add_exercise_inputs
    # %% Create prediction simulation
    sims = {}
    # set inputs accodring to data if it exists, otherwise to default value
    for idx, experiment in enumerate(data):
        ex = sund.Activity(time_unit='m')
        add_exercise_inputs(data[experiment]['input'],ex,addexercise=False)
        # create the activity based on the input data
        if 'percWmax_in' in activity.keys():
            exercisetime = np.linspace(-1,activity["percWmax_in"]['time'][-1]+10,2000) 
            ex.add_output("piecewise_constant", "percWmax_in", t=activity["percWmax_in"]['time'], f=activity["percWmax_in"]['values'])
            ex.add_output("constant", "watt_in", f=0)
        else:
            exercisetime = np.linspace(-1,activity["watt_in"]['time'][-1]+10,2000) 
            ex.add_output("constant", "percWmax_in", f=0)
            ex.add_output("piecewise_constant", "watt_in", t=activity["watt_in"]['time'], f=activity["watt_in"]['values'])
        sims[experiment] = sund.Simulation(models = model, activities = ex,
                                                time_unit='m',time_vector=exercisetime)

    return sims