import copy
import json
import numpy as np

ROUNDING_PRECISION = 13

def get_dof(data, do_validation=False):
    dof=0
    for experiment in data:
        if isinstance(data[experiment], dict) and 'input' not in data[experiment].keys():
            # Nested experiments share one parameter set.
            dof += get_dof(data[experiment],do_validation=do_validation)
        for feature in data[experiment].keys():
                if feature == 'SV' and 'SVi' in data[experiment].keys():
                    continue # SVi carries the same stroke-volume information when available.
                if feature == 'EDV' and 'EDVi' in data[experiment].keys():
                    continue # EDVi carries the same volume information when available.
                if feature == 'ESV' and 'ESVi' in data[experiment].keys():   
                    continue # ESVi carries the same volume information when available.

                if "time" not in data[experiment][feature]:
                    continue
                elif feature not in ['input','extra',"meta"] and (type(data[experiment][feature]["time"])==int or type(data[experiment][feature]["time"])==float) and not data[experiment][feature]["time"] is None:
                    dof += 1
                elif feature not in ['input','extra','meta'] and not data[experiment][feature]["time"] is None:
                    # During validation, exclude calibration points used to set initial conditions.
                    if do_validation and feature in ['SVi','EDVi','ESVi','SV','EDV','ESV','SVR']:
                        if data[experiment][feature]['mean'] is not None and data[experiment][feature]["time"][0] <= data[experiment]['input']['activitystarttime']:
                            if len(data[experiment][feature]['mean']) > 1:
                                dof += len(data[experiment][feature]["time"])-1
                            else:
                                continue
                    else:
                        dof += len(data[experiment][feature]["time"])
    return dof

def get_dof_subjects(data,do_validation=False):
    dof=0
    for subject in data:
        dof_s = get_dof(data[subject],do_validation=do_validation)
        dof += dof_s
    return dof


def load_litterature_data(filename='exercisedata_upright.json',validation=None,debug=False):
    from functions.utils import get_BSA

    with open("./data/"+filename, "r") as f:
        data_orig = json.load(f)

    # Remove extra info from data, and set SEM if it is missing
    data = copy.deepcopy(data_orig)
    # Standardize uncertainty fields before fitting.
    for experiment, d in data.items():
        d.pop('meta')
        featurelist = list(d.keys())
        featurelist.remove('input')
        
        for feature in featurelist:
            if 'sd' in d[feature] and d[feature]['sd'] is not None:
                d[feature]['SEM'] = np.divide(np.array(d[feature]['sd']),np.sqrt(d['input']['N']-1))
            elif  d[feature]['sem'] is None:
                # Use a conservative 10% relative SEM when no uncertainty is reported.
                d[feature]['SEM'] = np.multiply(0.1,d[feature]['mean'])
                d[feature].pop('sem')
                
            if 'sem' in d[feature]:
                d[feature]['SEM'] =copy.deepcopy(d[feature]['sem'])
                d[feature].pop('sem')

            # same datatype for time, mean and SEM
            if isinstance(d[feature]["mean"],int):
                d[feature]["mean"] = [d[feature]["mean"]]
                d[feature]["time"] = [int(d[feature]["time"])]
                if 'SEM' in d[feature]:
                    d[feature]["SEM"] = [int(d[feature]["SEM"])]
            elif isinstance(d[feature]["mean"],float):
                d[feature]["mean"] = [d[feature]["mean"]]
                d[feature]["time"] = [float(d[feature]["time"])]
                if 'SEM' in d[feature]:
                    d[feature]["SEM"] = [float(d[feature]["SEM"])]
        
        if 'height' in d['input'] and d['input']['height'] is not None and d['input']['height']>4:
            d['input']['height'] = d['input']['height']/100 # cm to m
        
        BSA = get_BSA(d['input'])

        # Convert stroke volume from L to mL when needed.
        if 'SV' in d and d['SV']['mean'] is not None and d['SV']['mean'][0]<5:
            d['SV']['mean'] = np.multiply(d['SV']['mean'],1000)
            d['SV']['SEM'] = np.multiply(d['SV']['SEM'],1000)
            if debug:
                print('converted stroke volume to mL')

        if 'SV' in featurelist and 'SVi' not in featurelist:
            d['SVi'] = {}
            d['SVi']['mean'] = copy.deepcopy(np.divide(d['SV']['mean'],BSA))
            mplus = np.divide(np.add(d['SV']['SEM'],d['SV']['mean']),BSA)
            d['SVi']['SEM'] = copy.deepcopy(np.subtract(mplus, d['SVi']['mean']))
            d['SVi']['time'] = copy.deepcopy(d['SV']['time'])
        if 'EDV' in featurelist and 'EDVi' not in featurelist:
            d['EDVi'] = {}
            d['EDVi']['mean'] = copy.deepcopy(np.divide(d['EDV']['mean'],BSA))
            mplus = np.divide(np.add(d['EDV']['SEM'],d['EDV']['mean']),BSA)
            d['EDVi']['SEM'] = copy.deepcopy(np.subtract(mplus, d['EDVi']['mean']))
            d['EDVi']['time'] = copy.deepcopy(d['EDV']['time'])
        if 'ESV' in featurelist and 'ESVi' not in featurelist:
            d['ESVi'] = {}
            d['ESVi']['mean'] = copy.deepcopy(np.divide(d['ESV']['mean'],BSA))
            mplus = np.divide(np.add(d['ESV']['SEM'],d['ESV']['mean']),BSA)
            d['ESVi']['SEM'] = copy.deepcopy(np.subtract(mplus, d['ESVi']['mean']))
            d['ESVi']['time'] = copy.deepcopy(d['ESV']['time'])
        if 'CO' in featurelist and 'CI' not in featurelist:
            d['CI'] = {}
            d['CI']['mean'] = copy.deepcopy(np.divide(d['CO']['mean'],BSA))
            mplus = np.divide(np.add(d['CO']['SEM'],d['CO']['mean']),BSA)
            d['CI']['SEM'] = copy.deepcopy(np.subtract(mplus, d['CI']['mean']))
            d['CI']['time'] = copy.deepcopy(d['CO']['time'])
        if 'VO2i' in featurelist and 'VO2' not in featurelist:
            d['VO2'] = {}
            d['VO2']['mean'] = copy.deepcopy(np.multiply(d['VO2i']['mean'],BSA))
            mplus = np.multiply(np.add(d['VO2i']['SEM'],d['VO2i']['mean']),BSA)
            d['VO2']['SEM'] = copy.deepcopy(np.subtract(mplus, d['VO2']['mean']))
            d['VO2']['time'] = copy.deepcopy(d['VO2i']['time'])

        # Convert VO2 from L/min to mL/min when needed.
        if 'VO2' in d and d['VO2']['mean'] is not None and d['VO2']['mean'][0]<100:
            d['VO2']['mean'] = np.multiply(d['VO2']['mean'],1000)
            d['VO2']['SEM'] = np.multiply(d['VO2']['SEM'],1000)
            if debug:
                print('converted vo2 py factor 1000 to ml')

        if 'RR' in featurelist and d['RR']['mean'] is not None:
            d['HR'] = {}
            hrmean =  np.divide(60,d['RR']['mean'])
            d['HR']['SEM'] = hrmean * np.divide(d['RR']['SEM'],d['RR']['mean'])
            d['HR']['mean'] = hrmean
            d['HR']['time'] = d['RR']['time']
            if debug:
                print('converted RR to bpm')
        
        # Set start time of exercise (needed to set initial conditions in the model later)
        if 'watt' in d['input'] and d['input']['watt'] is not None:
            inds = np.where(np.array(d['input']['watt']) != 0)
            d['input']['activitystarttime'] = d['input']['time'][inds[0][0]]
        elif 'percWmax' in d['input'] and d['input']['percWmax'] is not None:
            inds = np.where(np.array(d['input']['percWmax']) != 0)
            d['input']['activitystarttime'] = d['input']['time'][inds[0][0]]
        else:
            d['input']['activitystarttime'] = 0
        
        # Include event boundaries and measurement times so exercise inputs trigger correctly.
        t1 = d['input']['start']-1
        tend = d['input']['timeafter'] + d['input']['time'][-1]
        alltimepoints = [t for k, var in d.items() if k not in ["input", "meta", "extra"] for t in var["time"]]+[0]
        alltimepoints = alltimepoints + [t1] + [tend]

        timepoints_s = set(alltimepoints)
        timepoints = list(timepoints_s)
        timepoints.sort()
        d['input']['simulationtime'] = timepoints
        d['input']['temporal_res'] = 0.1 # minutes

        # Match each simulation time to the active workload level.
        d['input']['matching_percwatt'] = [None]*len(d['input']['simulationtime'])
        for idx,t in enumerate(d['input']['simulationtime']):
            if d['input']['percWmax'] is not None:
                smallw=[d['input']['percWmax'][i] for i,v in enumerate(d['input']['time']) if v<t]
                if len(smallw) == 0:
                    w=0
                else:
                    w = smallw[-1]
            else:
                smallw=[d['input']['watt'][i] for i,v in enumerate(d['input']['time']) if v<t]
                if len(smallw) == 0:
                    w = 0
                else:
                    wabs = smallw[-1]
                    if d['input']['wmax'] is None:
                        w = wabs/max(d['input']['watt'])+10
                    else:
                        w = wabs/d['input']['wmax']
            d['input']['matching_percwatt'][idx] = w

    # Prepare data for fitting to simulation
    data_fit = copy.deepcopy(data)
    if validation is not None:
        for val in validation:
            data_fit.pop(val)

    toremove = ['Qtot','Qlegs','Qarms','Qheadandtrunk','EF','VO2i','SV','EDV','ESV','CO','SaO2'] #these are not used

    for experiment in data_fit:
        featurelist = list(data_fit[experiment].keys())
        if 'percWmax' in featurelist:
            data_fit[experiment].pop('percWmax') # remove wmax since it is an input
        
        if 'watt' in featurelist:
            data_fit[experiment].pop('watt') # remove watt since it is an input

        if 'MAP' in featurelist and ('SBP' in featurelist and 'DBP' in featurelist):
            data_fit[experiment].pop('MAP') # remove MAP since it is a duplicate of data if both DBP and SBP exists

        for feat in toremove:
            if feat in featurelist:
                data_fit[experiment].pop(feat)

    
    # Prepare data for validation
    if validation is not None:
        data_val = {}
        for val in validation:
            if val in data:
                data_val[val] = copy.deepcopy(data[val])
        for experiment in data_val:
            featurelist = list(data_val[experiment].keys())
            if 'percWmax' in featurelist:
                data_val[experiment].pop('percWmax') # remove wmax since it is an input

            if 'watt' in featurelist:
                data_fit[experiment].pop('watt') # remove watt since it is an input

            if 'MAP' in featurelist and ('SBP' in featurelist and 'DBP' in featurelist):
                data_val[experiment].pop('MAP') # remove MAP since it is a duplicate of data if both DBP and SBP exists

            for feat in toremove:
                if feat in featurelist:
                    data_val[experiment].pop(feat) # remove
    else:
        data_val = {}

    return data_orig, data, data_fit, data_val


def calculate_average(data,variables, workloadtimes,subjects=None,modelname='exercise'):
    from functions.simfunctions import calc_wmax
    # average mechamp exercise data
    averages = {}
    averages['input'] = {}
    if modelname == 'exercise':
        variables.append('watt')
        variables.append('HRmax')
        variables.append('BSA')
        variables.append('activitystarttime')

    times = []
    if subjects is None:
        subjects = list(data.keys())
    for var in variables:
        if (modelname == 'exercise' and var in ['wmax','HRmax','BSA','EDV','ESV','EDVi','ESVi']):
            mean = [np.nan]
            time = [np.nan]
            SEM = [np.nan]
            inds = [workloadtimes[0]]
        else:
            inds = workloadtimes
            mean = [np.nan]*len(workloadtimes)
            time = [np.nan]*len(workloadtimes)
            SEM = [np.nan]*len(workloadtimes)
        alldata = {}
        for i in inds:
            n=0
            m=[]
            t=0
            if i > len(inds)-1:
                w=len(inds)-1
            else:
                w=i
            alldata[w] = []
            for subject in subjects:
                # find value at certain workload level
                if modelname == 'exercise':
                    worktime = data[subject]['input']['time'][w]
                    prevtime = worktime if w > 0 else -100.0
                    nexttime = data[subject]['input']['time'][w+1] if w < len(data[subject]['input']['time']) - 1 else worktime+100
                if var[0:3] == 'COi' and var not in data[subject]:
                    HR = np.array(data[subject]['HR']['mean'])
                    SVtime = np.array(data[subject]['SVi']['time'])
                    HRtime = np.array(data[subject]['HR']['time'])
                    HR = np.interp(SVtime, HRtime, HR)  # Interpolate HR to SV time points
                    subjmean = np.divide(np.multiply(np.array(HR),np.array(data[subject]['SVi']['mean'])), 1000)  # Convert to L/min
                    subjtime = np.array(data[subject]['SVi']['time'])
                elif var[0:2] == 'CO' and var not in data[subject]:
                    HR = np.array(data[subject]['HR']['mean'])
                    SVtime = np.array(data[subject]['SV']['time'])
                    HRtime = np.array(data[subject]['HR']['time'])
                    HR = np.interp(SVtime, HRtime, HR)  # Interpolate HR to SV time points
                    subjmean = HR * np.array(data[subject]['SV']['mean']) / 1000  # Convert to L/min
                    subjtime = np.array(data[subject]['SV']['time'])
                elif var[-4:] == 'perc':
                    subjmean = data[subject][ var[:-4]]['mean']
                    subjtime = data[subject][ var[:-4]]['time']
                elif var == 'watt': #get workload to model input
                    subjmean = data[subject]['input']['watt']
                    subjtime = data[subject]['input']['time']
                elif var == 'wmax': #get workload max to model input
                    subjmean = calc_wmax(data[subject])
                    subjtime = 0
                elif var == 'percWmax': #get workload to model input
                    wmax = calc_wmax(data[subject])
                    subjmean = 100*np.array(data[subject]['input']['watt']) / wmax if wmax > 0 else 0
                    subjtime = data[subject]['input']['time']
                elif var in ['BSA','activitystarttime','MRItimes']:
                    subjmean = data[subject]['input'][var]
                    subjtime = data[subject]['input'][var]
                elif var == 'HRmax':
                    age = data[subject]['input']['age']
                    subjmean = (220-age)*0.85
                    subjtime = 0
                else:
                    subjmean = data[subject][var]['mean']
                    subjtime = data[subject][var]['time']
                
                if var[-4:] == 'perc':
                    idx = np.where(np.array(subjtime) < data[subject]['input']['activitystarttime'])[0]
                    restval = np.mean(np.array(subjmean)[idx]) if len(idx) > 0 else None
                    subjmean = (np.array(subjmean)-restval)/restval if restval is not None else None
                    subjtime = subjtime

                if subjmean is not None:
                    if isinstance(subjmean,list) or not isinstance(subjmean,(float,int)):
                        idx = np.where((np.array(subjtime) >= prevtime) & (np.array(subjtime) < nexttime))[0]
                        if idx.size > 0:
                            vals = np.array(subjmean)[idx]
                            vals = vals[~np.isnan(vals)]
                            m.append(np.mean(vals))
                            t +=  np.mean(np.array(subjtime)[idx])
                            n+=1
                        else:
                            m.append(np.nan)
                    elif subjmean is not None and ~np.isnan(subjmean):
                        m.append(np.mean(np.array(subjmean)))
                        t +=  np.mean(np.array(subjtime))
                        n+=1
                    else:
                        m.append(np.nan)
                else:
                    m.append(np.nan)
            m = np.array(m)
            if n > 0:
                alldata[w] = m
                mean[w] = sum(m[~np.isnan(m)]) / n
                time[w] = t / n
                SEM[w] = np.std(m[~np.isnan(m)]) / np.sqrt(n)

        if var in ['HRmax','wmax','BSA','activitystarttime']:
            averages['input'][var] = mean[0]
            averages['input'][var + 'SEM'] = SEM[0]
            averages['input'][var + 'ALL'] = alldata
        elif var in ['percWmax','watt','MRItimes']:
            averages['input'][var] = mean
            averages['input'][var + 'SEM'] = SEM
            averages['input'][var + 'ALL'] = alldata
            averages['input']['time'] = time
        else:
            nanind = ~np.isnan(mean) & ~np.isnan(time)
            averages[var] = {
                'mean': np.array(mean)[nanind],
                'time': np.array(time)[nanind],
                'SEM': np.array(SEM)[nanind],
                'alldata' : alldata
            }
            times = np.append(times,time)

    if 'time' in averages['input']:
        times = np.append(times,averages['input']['time'])

    averages['input']['simulationtime'] = np.unique(times[~np.isnan(times)])
    averages['input']['temporal_res'] = 0.1

    #get workload to model input. Base on the avearage HR and estimated HRmax average and given watt
    if 'watt' in averages['input']:
        wmax = calc_wmax(averages)
        averages['input']['percWmax'] = 100*np.array(averages['input']['watt']) / wmax
        averages['input']['wmax'] = wmax

    return averages