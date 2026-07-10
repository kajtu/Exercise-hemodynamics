import matplotlib.pyplot as plt
import numpy as np
from functions.simfunctions import avatar_simulation,exercise_simulation
import copy, os, json
from functions.utils import colorpicker, hex_to_rgb

def plot_exercise_times(ax, values, time,heartbeattimes=[],heartbeatcols=None,heartbeatnames=None,printvalue=False,fc='grey',ls='--',alphaval = 0.5): 
    maxval = np.max(np.array(values))
    # if there is only one value, make it a bit lighter color
    if not np.any((np.array(values) > 0) & (np.array(values) < maxval)):
        alphaval = 0.3
    start_time = 0
    for i in range(1, len(values)):
        if values[i]>0 and values[i - 1] != values[i]:
            if values[i - 1] != 0:
                end_time = time[i]
                ax.axvspan(start_time, end_time, fc=fc, alpha=alphaval*(values[i-1]/maxval),ec=None)
                if printvalue:
                    # write out the values on top
                    step=0.01*(plt.ylim()[1]-plt.ylim()[0])
                    ax.annotate(f'{100*values[i-1]/maxval:.0f}%', xy =(start_time, plt.ylim()[0]+step),fontsize=7)
            start_time = time[i]
        elif values[i] == 0 and values[i - 1] != 0:
            end_time = time[i]
            ax.axvspan(start_time, end_time, fc=fc, alpha=alphaval*(values[i-1]/maxval),ec=None)
            if printvalue:
                # write out the values on top
                step=0.01*(plt.ylim()[1]-plt.ylim()[0])
                ax.annotate(f'{100*values[i-1]/maxval:.0f}%', xy =(start_time, plt.ylim()[0]+step),fontsize=7)
            
    if values[-1]>0:
        end_time = time[-1]
        ax.axvspan(start_time, end_time, fc=fc, alpha=alphaval*(values[i-1]/maxval),ec=None)
        if printvalue:
            # write out the values on top
            step=0.01*(plt.ylim()[1]-plt.ylim()[0])
            ax.annotate(f'{100*values[i-1]/maxval:.0f}%', xy =(start_time, plt.ylim()[0]+step),fontsize=7)



    # Plot lines for the different MRI measurements
    for i,t in enumerate(heartbeattimes):
        if heartbeatcols is not None:
            ax.axvline(x=t, color=heartbeatcols[i], linestyle=ls)
        else:
            ax.axvline(x=t, color='r', linestyle=ls)


def plot_sim(params, sim, timepoints,c='b', feature_to_plot='SBP',secondfeature = None,model=None,datatime=None,lw=1,ls='-',label='',simSS=None,ax=None,T=1):
    if model is None:
        # Plot only
        feature_idx = sim.feature_names.index(feature_to_plot)
        if secondfeature is None and sim.feature_data is not None:
            if ax is None:
                plt.plot(np.array(sim.time_vector)/T, sim.feature_data[:,feature_idx], color=c,lw=lw,linestyle=ls,label=label)
            else:
                ax.plot(np.array(sim.time_vector)/T, sim.feature_data[:,feature_idx], color=c,lw=lw,linestyle=ls,label=label)
        elif sim.feature_data is not None:
            feature_idx2 = sim.feature_names.index(secondfeature)
            if ax is None:
                plt.plot(sim.feature_data[:,feature_idx], sim.feature_data[:,feature_idx2], color=c,lw=lw,linestyle=ls,label=label)
            else:
                ax.plot(sim.feature_data[:,feature_idx], sim.feature_data[:,feature_idx2], color=c,lw=lw,linestyle=ls,label=label)
        if datatime is not None and feature_idx is not None:
            y_sim = copy.deepcopy(sim.feature_data[:,feature_idx])
            y_sim = y_sim[np.searchsorted(sim.time_vector, datatime)]
            y_t = sim.time_vector[np.searchsorted(sim.time_vector, datatime)]
            if ax is None:
                plt.plot(np.array(y_t)/T, y_sim, 'o',fillstyle='none',ms=5,color=c,label=label)
            else:
                ax.plot(np.array(y_t)/T, y_sim, 'o',fillstyle='none',ms=5,color=c,label=label)
    else:
        # Simulate
        try:
            if model == 'exercise':
                exercise_simulation(params, sim,simSS,timepoints=timepoints)
            elif model == 'avatar':
                if params is None:
                    print('No params given for avatar simulation in plot_sim!')
                else:
                    avatar_simulation(params,sim,simSS,timepoints=timepoints)
            else:
                sim.simulate(parameter_values=params,time_vector=timepoints,reset=True)
        except Exception as e:
            print('simulation failed in plot_sim: ', e)
            return
        # Plot the selected feature
        feature_idx = sim.feature_names.index(feature_to_plot)
        if secondfeature is None and sim.feature_data is not None:
            if ax is None:
                plt.plot(np.array(sim.time_vector)/T, sim.feature_data[:,feature_idx], color=c,lw=lw,linestyle=ls,label=label)
            else:
                ax.plot(np.array(sim.time_vector)/T, sim.feature_data[:,feature_idx], color=c,lw=lw,linestyle=ls,label=label)
        elif sim.feature_data is not None:
            feature_idx2 = sim.feature_names.index(secondfeature)
            if ax is None:
                plt.plot(sim.feature_data[:,feature_idx], sim.feature_data[:,feature_idx2],color=c,lw=lw,linestyle=ls,label=label)
            else:
                ax.plot(sim.feature_data[:,feature_idx], sim.feature_data[:,feature_idx2],color=c,lw=lw,linestyle=ls,label=label)

def prediction_max_exercise(params, data,model,modelname,features={'CI','MAP','R'},
                            activity = {'percWmax_in':{'values':[0,55,70,85,100,0],'time':[0,2,4,6,8]}},
                            figname='Prediction_maxexercise',do_stattest = True,savefig=False,
                            subjects=None,uncname=''):
    from functions.simfunctions import add_exercise_inputs
    import sund
    from scipy.stats import ttest_ind

    # %% Create prediction simulation
    sims = {}
    simsSS={}
    # set inputs accodring to data if it exists, otherwise to default value
    for idx, experiment in enumerate(data):
        ex = sund.Activity(time_unit='m')
        add_exercise_inputs(data[experiment]['input'],ex,addexercise=False)
        # create the activity based on the input data
        if 'percWmax_in' in activity.keys():
            exercisetime = np.linspace(-1,activity["percWmax_in"]['time'][-1]+10,2000) #activity["percWmax_in"]['time'][-1]+60
            ex.add_output(type="piecewise_constant", name="percWmax_in", t=activity["percWmax_in"]['time'], f=activity["percWmax_in"]['values'])
            ex.add_output(type="constant", name="watt_in", f=0)
            endtime = activity["percWmax_in"]['time'][-1]
        else:
            exercisetime = np.linspace(-1,activity["watt_in"]['time'][-1]+10,2000) #activity["percWmax_in"]['time'][-1]+60
            ex.add_output(type="constant", name="percWmax_in", f=0)
            ex.add_output(type="piecewise_constant", name="watt_in", t=activity["watt_in"]['time'], f=activity["watt_in"]['values'])
            endtime = activity["watt_in"]['time'][-1]
        sims[experiment] = sund.Simulation(models = model, activities = ex,
                                                time_unit='m',time_vector=exercisetime)
        simsSS[experiment] = None # exercise model is already in ss      

    # load UC results
    UC_data = {}
    if os.path.exists(f"parameters/{model.name}/UC/UC_PPL_{model.name}_pred.json"):
        with open(f"parameters/{model.name}/UC/UC_PPL_{model.name}_pred.json", 'r') as file:
            UC_data = json.load(file)
        UC_data = UC_data[uncname] if uncname in UC_data else {}
    # %% Plot prediction for each subj/exp        
    #nrows = 2
    if subjects is None:
        subjects = list(sims.keys())
    ncols=2
    nrows = int(np.ceil(len(features)/ncols))
    colors = plt.cm.Set2(np.linspace(0, 1, len(subjects)))
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figname,figsize=((20/2)*cm, 4.5*nrows*cm),layout="constrained")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
    plotnum = 1
    for idx, feature in enumerate(features):
        ax=plt.subplot(nrows,ncols,plotnum)
        plotnum += 1
        resting_pats=[]
        resting_conts=[]
        max_pats = []
        max_conts = []
        for jx, experiment in enumerate(subjects):
            if experiment in sims and feature in sims[experiment].feature_names:
                if UC_data and experiment in UC_data and feature in UC_data[experiment] and UC_data[experiment][feature]["Min"] is not None and UC_data[experiment][feature]["Min"][0] is not None:
                    ax.fill_between(np.array(UC_data[experiment]["Time"]),
                                    UC_data[experiment][feature]["Min"],
                                    UC_data[experiment][feature]["Max"],
                                    alpha=0.2, color=colorpicker(experiment,backupcolor=colors[jx]))
                if idx == 0:
                    plot_sim(params[experiment], sims[experiment], [],c=colorpicker(experiment,backupcolor=colors[jx]), feature_to_plot=feature,lw=1.5,label=experiment,model=modelname,simSS=simsSS[experiment])
                else:
                    plot_sim(params[experiment], sims[experiment], [],c=colorpicker(experiment,backupcolor=colors[jx]), feature_to_plot=feature,lw=1.5,model=modelname,simSS=simsSS[experiment])
                if experiment[0] == 'p':
                    resting_pats.append(sims[experiment].feature_data[0,sims[experiment].feature_names.index(feature)])
                    if feature in ['SVR','SVR change','ESVi','ESV change']: # features that are decreasing with exercise
                        max_pats.append(min(sims[experiment].feature_data[:,sims[experiment].feature_names.index(feature)]))
                    else:
                        max_pats.append(max(sims[experiment].feature_data[:,sims[experiment].feature_names.index(feature)]))
                elif experiment[0] == 'c':
                    resting_conts.append(sims[experiment].feature_data[0,sims[experiment].feature_names.index(feature)])
                    if feature in ['SVR','SVR change','ESVi','ESV change']: #features that are decreasing with exercise
                        max_conts.append(min(sims[experiment].feature_data[:,sims[experiment].feature_names.index(feature)]))
                    else:
                        max_conts.append(max(sims[experiment].feature_data[:,sims[experiment].feature_names.index(feature)]))

            
            elif experiment[0] == 'p':
                resting_pats.append(np.nan)
                if feature in ['SVR','SVR change','ESVi','ESV change']:
                    max_pats.append(np.nan)
            elif experiment[0] == 'c':
                resting_conts.append(np.nan)
                if feature in ['SVR','SVR change','ESVi','ESV change']:
                    max_conts.append(np.nan)
            if jx == 0 and experiment in sims:
                wattsim = sims[experiment].feature_data[:,sims[experiment].feature_names.index('percWmax')]
                plot_exercise_times(ax, wattsim, sims[experiment].time_vector,printvalue=False)
            #if feature == 'HR':
            #    ax.axhline(y=sims[experiment].feature_data[-1,sims[experiment].feature_names.index('HRmax')], color=colorpicker(experiment,backupcolor=colors[jx]), linestyle='--', lw=0.5)
            if idx == len(features)-1:
                plt.xlabel('Time (min)')
            plt.ylabel(feature + ' (' + model.feature_units[model.feature_names.index(feature)] + ')')
        
        # plot mean resting value for controls vs patients
        if do_stattest and len(resting_pats) > 0 and len(resting_conts) > 0 and feature[-6:] != 'change':
            t_stat, p_val = ttest_ind(resting_pats, resting_conts)
            patcol=hex_to_rgb(colorpicker('p',backupcolor=colors[jx]))*0.8
            ccol=hex_to_rgb(colorpicker('c',backupcolor=colors[jx]))*0.8
            plt.errorbar(-8,np.mean(resting_pats), yerr=np.std(resting_pats), color=patcol, marker='o', markersize=5,capsize=3)
            plt.errorbar(-7,np.mean(resting_conts), yerr=np.std(resting_conts), color=ccol, marker='o', markersize=5,capsize=3)
            ax.axhline(y=np.mean(resting_pats), color=patcol, linestyle='--', lw=1)
            ax.axhline(y=np.mean(resting_conts), color=ccol, linestyle='--', lw=1)
            if p_val < 0.001:
                plt.text(-19, np.mean([np.mean(resting_pats), np.mean(resting_conts)])*0.9, f'p <\n0.001*', color='black', fontsize=7)
            elif p_val < 0.05:
                plt.text(-19, np.mean([np.mean(resting_pats), np.mean(resting_conts)])*0.9, f'p =\n{p_val:.3f}*', color='black', fontsize=7)
            else:
                plt.text(-19, np.mean([np.mean(resting_pats), np.mean(resting_conts)])*0.9, f'p =\n{p_val:.3f}', color='black', fontsize=7)
            plt.xlim(-20,plt.xlim()[1])
        if do_stattest and len(max_pats) > 0 and len(max_conts) > 0:
            t_stat, p_val = ttest_ind(max_pats, max_conts)
            patcol=hex_to_rgb(colorpicker('p',backupcolor=colors[jx]))*0.8
            ccol=hex_to_rgb(colorpicker('c',backupcolor=colors[jx]))*0.8
            plt.errorbar(-4,np.mean(max_pats), yerr=np.std(max_pats), color=patcol, marker='^', markersize=5,capsize=3)
            plt.errorbar(-3,np.mean(max_conts), yerr=np.std(max_conts), color=ccol, marker='^', markersize=5,capsize=3)
            ax.axhline(y=np.mean(max_pats), color=patcol, linestyle=':', lw=1)
            ax.axhline(y=np.mean(max_conts), color=ccol, linestyle=':', lw=1)
            if p_val < 0.001:
                plt.text(-19, max([np.mean(max_pats), np.mean(max_conts)]), f'p <\n0.001*', color='black', fontsize=7)
            elif p_val < 0.05:
                plt.text(-19, max([np.mean(max_pats), np.mean(max_conts)]), f'p =\n{p_val:.3f}*', color='black', fontsize=7)
            else:
                plt.text(-19, max([np.mean(max_pats), np.mean(max_conts)]), f'p =\n{p_val:.3f}', color='black', fontsize=7)
            plt.xlim(-20,plt.xlim()[1])
        plt.xticks([0,endtime,endtime+10])

    fig.legend(loc='outside upper center',ncols=10)

    if savefig:
        fig.savefig(f"./figures/{figname}.png", dpi=300)
        fig.savefig(f"./figures/{figname}.svg")

def get_pvaluetext(x,y,independent=True,parametric=True,significantdigits=3):
    from scipy.stats import ttest_ind, ttest_rel,mannwhitneyu,wilcoxon
    if independent and parametric:
        t_stat, p_val = ttest_ind(x, y)
    elif not independent and parametric:
        t_stat, p_val = ttest_rel(x, y)
    elif independent and not parametric:
        t_stat, p_val = mannwhitneyu(x, y)
    elif not independent and not parametric:
        t_stat, p_val = wilcoxon(x, y)
    
    if p_val < 10**-significantdigits:
        ptext = f'p<{10**-significantdigits}*'
    elif p_val < 0.05:
        ptext = f'p={p_val:.{significantdigits}f}*'
    else:
        ptext = f'p={p_val:.{significantdigits}f}'
    return ptext

def plot_change_between_sims(features,subjects,sims,expnames=['Rest','Exercise'],figname='sim_rest_vs_exercise',savefig=False,percent=False,parametric=False,uncnames={}):
    if percent:
        ncols=8
        rowheight=4.5
        figwidth=18
    else:
        ncols = 4
        rowheight=3.5
        figwidth=17
    nrows = int(np.ceil(len(features)/ncols))
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figname,figsize=(figwidth*cm, rowheight*nrows*cm),layout="constrained")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plotnum = 1 

    for feature in features:
        pvals=[]
        cvals=[]
        pvals_e=[]
        cvals_e=[]
        fname = features[feature][1]
        funit = features[feature][2]
        fminmax = features[feature][0]
        ax = plt.subplot(nrows,ncols,plotnum)
        plotnum += 1
        for si,s in enumerate(subjects):
            # load UC results for this subject
            UC_data = {}
            if os.path.exists(f'parameters/avatar_exercise/UC/UC_PPL_{s}.json'):
                with open(f'parameters/avatar_exercise/UC/UC_PPL_{s}.json', 'r') as file:
                    UC_data = json.load(file) #contains both rest and exercise
            if percent:
                xvals=0
            else:
                xvals = np.array([1,2]) 
            if s[0] == 'p':
                if percent: 
                    xvals=xvals+len(expnames)
                else:
                    xvals = np.array([2.5,3.5]) 
            vals = [np.nan]*len(expnames)
            for e,exp in enumerate(expnames):
                if fname in sims[s][exp].feature_names and sims[s][exp].feature_data is not None:
                    i = sims[s][exp].feature_names.index(fname)
                    if fminmax == 'max':
                        vals[e] = max(sims[s][exp].feature_data[:,i])
                    elif fminmax == 'min':
                        vals[e] = min(sims[s][exp].feature_data[:,i])
                    elif fminmax == 'mean':
                        vals[e] = np.mean(sims[s][exp].feature_data[:,i])
            if percent:
                vals = (vals[1] - vals[0])/vals[0]*100
                plt.plot(xvals ,vals,marker='o',markersize=4,c=colorpicker(s,backupcolor='k'))
                if s[0] == 'p':
                    pvals.append(vals)
                else:
                    cvals.append(vals)
            else:
                plt.plot(xvals,vals,marker='o',c=colorpicker(s,backupcolor='k'))
                # plot uncertainty if available
                for e,exp in enumerate(expnames):
                    if uncnames and feature in uncnames:
                        ucname = uncnames[feature]
                    elif fname == 'Elv':
                        ucname = 'Emax_LV'
                    elif fname == 'Ela':
                        ucname = 'Emax_LA'
                    else:
                        ucname = fname
                    if UC_data and exp in UC_data and ucname in UC_data[exp] and UC_data[exp][ucname]['Max'] is not None:
                        if isinstance( UC_data[exp][ucname]['Max'],list) and UC_data[exp][ucname]['Max'][0] is not None:
                            if fminmax == 'max':
                                lb = max(UC_data[exp][ucname]['Min'])
                                ub = max(UC_data[exp][ucname]['Max'])
                            elif fminmax == 'min':
                                lb = min(UC_data[exp][ucname]['Min'])
                                ub = min(UC_data[exp][ucname]['Max'])
                            elif fminmax == 'mean':
                                lb = np.mean(UC_data[exp][ucname]['Min'])
                                ub = np.mean(UC_data[exp][ucname]['Max'])
                        else:
                            lb = UC_data[exp][ucname]['Min']
                            ub = UC_data[exp][ucname]['Max']
                        plt.plot([xvals[e],xvals[e]],[lb,ub],c=colorpicker(s,backupcolor='k'),lw=1)
                if s[0] == 'p':
                    pvals.append(vals[0])
                    pvals_e.append(vals[1])
                else:
                    cvals.append(vals[0])
                    cvals_e.append(vals[1])
        pvals = np.array(pvals)
        cvals = np.array(cvals)
        pvals_e = np.array(pvals_e)
        cvals_e = np.array(cvals_e)
        if percent:
            ttest_result=get_pvaluetext(pvals[~np.isnan(pvals)], cvals[~np.isnan(cvals)],independent=True,parametric=parametric,significantdigits=3)
            plt.errorbar(-0.25,np.mean(cvals[~np.isnan(cvals)]), yerr=np.std(cvals[~np.isnan(cvals)]), color=colorpicker('average_control'), marker='o', markersize=5,capsize=3)
            plt.errorbar(len(expnames)+0.25,np.mean(pvals[~np.isnan(pvals)]), yerr=np.std(pvals[~np.isnan(pvals)]), color=colorpicker('average_pat'), marker='o', markersize=5,capsize=3)
            plt.title(ttest_result, fontsize=8)
            plt.ylabel(f"{feature} (% change)")
            plt.axhline(y=0,c='k',ls='--',lw=0.5)
            plt.xticks([0,len(expnames)],['C','HT'])
            plt.xlim(-1,len(expnames)+1)
        else:
            plt.ylim(max(plt.ylim()[0],np.mean(pvals[~np.isnan(pvals)])*0.001),min(plt.ylim()[1]*1.1,np.mean(pvals[~np.isnan(pvals)])*10))
            #hypertensive
            nanmask = ~np.isnan(pvals) & ~np.isnan(pvals_e)
            ttest_result=get_pvaluetext(pvals[nanmask], pvals_e[nanmask],independent=False,parametric=parametric,significantdigits=3)
            plt.text(3, plt.ylim()[1]*0.9, ttest_result, fontsize=8, ha='center')
            #controls
            nanmask = ~np.isnan(cvals) & ~np.isnan(cvals_e)
            ttest_result=get_pvaluetext(cvals[nanmask], cvals_e[nanmask],independent=False,parametric=parametric,significantdigits=3)
            plt.text(1.5, plt.ylim()[1]*0.9, ttest_result, fontsize=8, ha='center')
            # fix labels and axes
            plt.ylabel(f"{feature} ({funit})")
            expnamesSHORT = [ename[0] for ename in expnames]
            plt.xticks([1,2,2.5,3.5],expnamesSHORT+expnamesSHORT,rotation=45)
            plt.xlim(0.75,3.75)
    if savefig:
        # create folder if it doesn't exist
        fig.savefig(f"./figures/{figname}.png", dpi=300)
        fig.savefig(f"./figures/{figname}.svg")

def calc_percchange_table(features,subjects,sims,expnames=['Rest','Exercise'],savetables=False,parametric=False,uncnames={}):
    import pandas as pd
    from scipy.stats import ttest_ind
        # features={'EmaxLV':['max','Elv','mmHg/mL'], 'EmaxLA':['max','Ela','mmHg/mL'],
    #'Central SBP':['max','Pao','mmHg'],'Central DBP':['min','Pao','mmHg'],'Central MAP':['mean','Pao','mmHg'
    change_dict = {'vals':['Control', 'HT','p-value','Individual SD (mean [range])']}
    for feature in features:
        fname = features[feature][1]
        funit = features[feature][2]
        fminmax = features[feature][0]
        pvals=[]
        cvals=[]
        sds = []
        for si,s in enumerate(subjects):
            # load UC results for this subject
            UC_data = {}
            if os.path.exists(f'parameters/avatar_exercise/UC/UC_PPL_{s}.json'):
                with open(f'parameters/avatar_exercise/UC/UC_PPL_{s}.json', 'r') as file:
                    UC_data = json.load(file) #contains both rest and exercise
            vals = [np.nan]*len(expnames)
            for e,exp in enumerate(expnames):
                if fname in sims[s][exp].feature_names and sims[s][exp].feature_data is not None:
                    i = sims[s][exp].feature_names.index(fname)
                    if fminmax == 'max':
                        vals[e] = max(sims[s][exp].feature_data[:,i])
                    elif fminmax == 'min':
                        vals[e] = min(sims[s][exp].feature_data[:,i])
                    elif fminmax == 'mean':
                        vals[e] = np.mean(sims[s][exp].feature_data[:,i])
            vals = (vals[1] - vals[0])/vals[0]*100
            if s[0] == 'p':
                pvals.append(vals)
            else:
                cvals.append(vals)

            # uncertainty
            lb=1e6
            ub=-1e6
            if feature in uncnames:
                fnameunc = uncnames[feature]
            else:
                fnameunc = ''
            # check if really should load from both rest and exercise - in rest it is always 0 or?
            for e,exp in enumerate(expnames):
                if UC_data and exp in UC_data and fnameunc in UC_data[exp] and UC_data[exp][fnameunc]['Max'] is not None:
                    if isinstance(UC_data[exp][fnameunc]['Max'], list) and UC_data[exp][fnameunc]['Max'][0] is not None:
                        lbi = min(UC_data[exp][fnameunc]['Min'])
                        ubi = max(UC_data[exp][fnameunc]['Max'])
                    else:
                        lbi = UC_data[exp][fnameunc]['Min']
                        ubi = UC_data[exp][fnameunc]['Max']
                    lb = min(lb,lbi)
                    ub = max(ub,ubi)
            if lb == 1e6 or ub == -1e6:
                sds.append(np.nan)
            else:
                # find standard deviation of the parameter for this subject
                middlep = lb+ ((ub-lb)/2) #mid point of the confidence interval
                sd = (ub-middlep) / 1.96; #95% conf interval corresponds to 1.96 sd
                sds.append(sd)
            
        sds = np.array(sds)
        if len(sds[~np.isnan(sds)])>0:
            sds = sds[~np.isnan(sds)]
        cvals = np.array(cvals)
        pvals = np.array(pvals)
        ttest_result=get_pvaluetext(pvals[~np.isnan(pvals)], cvals[~np.isnan(cvals)],independent=True,parametric=parametric,significantdigits=4)
        standarddev = f'{np.mean(sds):.1f} [{np.min(sds):.1f}-{np.max(sds):.1f}]'
        change_dict[feature + ' (' + funit + ')'] = [f'{np.mean(cvals[~np.isnan(cvals)]):.1f}+-{np.std(cvals[~np.isnan(cvals)]):.1f}', 
                                                     f'{np.mean(pvals[~np.isnan(pvals)]):.1f}'+f'+-{np.std(pvals[~np.isnan(pvals)]):.1f}',
                                                     ttest_result,standarddev]
    dfchange = pd.DataFrame(change_dict)
    dfchange= dfchange.T
    print(dfchange)
    if savetables:
        dfchange.to_excel("./tables/Table7_simulated_change_withexercise.xlsx")
    return dfchange
 

def table_paramvalues(features,subjects,sims,savetables=False,parametric=False,tablename = 'featurevalues',modelname='bp_exercise_fb_both_rand'):
    import pandas as pd

    change_dict = {'vals':['Control (mean+-sd [range])', 'HT (mean+-sd [range])','p-value','Individual SD (mean [range])']}
    # load UC results 
    UC_data = {}
    if os.path.exists(f'parameters/{modelname}/UC/UC_PPL_{modelname}.json'):
        with open(f'parameters/{modelname}/UC/UC_PPL_{modelname}.json', 'r') as file:
            UC_data = json.load(file) #contains both rest and exercise

    for feature in features:
        fname = features[feature][1]
        funit = features[feature][2]
        fminmax = features[feature][0]
        pvals=[]
        cvals=[]
        sds = []
        vals=None
        for si,s in enumerate(subjects):
            if fname in sims[s].feature_names and sims[s].feature_data is not None:
                i = sims[s].feature_names.index(fname)
                if fminmax == 'max':
                    vals = max(sims[s].feature_data[:,i])
                elif fminmax == 'min':
                    vals = min(sims[s].feature_data[:,i])
                elif fminmax == 'mean':
                    vals = np.mean(sims[s].feature_data[:,i])
            if s[0] == 'p':
                pvals.append(vals)
            else:
                cvals.append(vals)

            # uncertainty
            lb=1e6
            ub=-1e6
            # check if really should load from both rest and exercise - in rest it is always 0 or?
            if UC_data and s in UC_data and feature in UC_data[s]:
                lbi = min(UC_data[s][feature]['Min'])
                ubi = max(UC_data[s][feature]['Max'])
                lb = min(lb,lbi)
                ub = max(ub,ubi)
            if lb == 1e6 or ub == -1e6:
                sds.append(np.nan)
            else:
                # find standard deviation of the parameter for this subject
                middlep = lb+ ((ub-lb)/2) #mid point of the confidence interval
                sd = (ub-middlep) / 1.96; #95% conf interval corresponds to 1.96 sd
                sds.append(sd)
            
        sds = np.array(sds)
        if len(sds[~np.isnan(sds)])>0:
            sds = sds[~np.isnan(sds)]
        cvals = np.array(cvals)
        pvals = np.array(pvals)
        if feature in ['keR_r','keESV_r']:
            pvals = pvals*100 # convert to percent
            cvals = cvals*100
            sds = sds*100
        ttest_result=get_pvaluetext(pvals[~np.isnan(pvals)], cvals[~np.isnan(cvals)],independent=True,parametric=parametric,significantdigits=4)
        standarddev = f'{np.mean(sds):.1f} [{np.min(sds):.1f}-{np.max(sds):.1f}]'
        change_dict[feature + ' (' + funit + ')'] = [f'{np.mean(cvals[~np.isnan(cvals)]):.1f}+-{np.std(cvals[~np.isnan(cvals)]):.1f} [{np.min(cvals[~np.isnan(cvals)]):.1f}-{np.max(cvals[~np.isnan(cvals)]):.1f}]', 
                                                     f'{np.mean(pvals[~np.isnan(pvals)]):.1f}'+f'+-{np.std(pvals[~np.isnan(pvals)]):.1f}  [{np.min(pvals[~np.isnan(pvals)]):.1f}-{np.max(pvals[~np.isnan(pvals)]):.1f}]',
                                                     ttest_result,standarddev]
    dfchange = pd.DataFrame(change_dict)
    dfchange= dfchange.T
    print(dfchange)
    if savetables:
        dfchange.to_excel(f"./tables/{tablename}_{modelname}.xlsx")
    return dfchange