import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
from functions.simfunctions import estimate_sbp
import copy
from functions.simulation_plot_functions import plot_exercise_times,plot_sim
from functions.utils import colorpicker
import json
import os

# Define a function to plot one feature in one dataset
def plot_datafeature(data,featurename,c='k',marker='o',lw=1,cz=3,ms=3,ax=None,T=1):
    if  not data['mean'] is None:
        if  data['SEM'] is None or cz == 0:
            if ax is None:
                plt.plot(np.array(data['time'])/T, data['mean'], marker=marker, color=c,ms=ms,lw=lw)
            else:
                ax.plot(np.array(data['time'])/T, data['mean'], marker=marker, color=c,ms=ms,lw=lw)
        else:
            if ax is None:
                plt.errorbar(np.array(data['time'])/T, data['mean'],  data['SEM'],linestyle='None', marker=marker, color=c,capsize=cz,ms=ms,lw=lw)
            else:
                ax.errorbar(np.array(data['time'])/T, data['mean'],  data['SEM'],linestyle='None', marker=marker, color=c,capsize=cz,ms=ms,lw=lw)
        return True
    return False

# Define a function to plot  feature vs feature in one dataset
def plot_two_datafeatures(data,featurename1,featurename2,c='k'):
    if  data[featurename1]['time'] == data[featurename2]['time'] and not data[featurename1]['mean'] is None and not data[featurename2]['mean'] is None:
        if  data[featurename1]['SEM'] is None or data[featurename2]['SEM'] is None:
            plt.plot(data[featurename1]['mean'], data[featurename2]['mean'], marker='o', color=c,ms=2)
        else:
            plt.errorbar(data[featurename1]['mean'], data[featurename2]['mean'], xerr=data[featurename1]['SEM'],yerr=data[featurename2]['SEM'],linestyle='None', marker='o', color=c,capsize=2,ms=2)
    plt.ylabel(featurename1)
    plt.ylabel(featurename2)


def plot_avatar_all(params=None,sims=None,simsSS=None,data=None,figname='All_subjects_heartbeats',
                    savefigs=False,dofigure=True,nrows=None,pindex=0,allinone=False,subjin=None,percent=False):
    # Assuming that simulations are already performed outside of this function
    labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
    features =('MV','AV','AA') 
    cols = plt.cm.Set2(np.linspace(0, 1, 8))
    cols  = np.vstack((cols,cols))#repeat the same colors twice in case of more than 8 needed
    if sims is not None:
        subjects = list(sims.keys())
    elif data is not None:
        subjects = list(data.keys())
    else:
        print('no data or simulations provided')
        return
    if dofigure:
        cm = 1/2.54  # centimeters in inches
        fig=plt.figure(figname,figsize=(20*cm, min(len(subjects)*3,25)*cm),layout="constrained")
        plt.rcParams.update({'font.family': 'serif','font.size': 7})
        if nrows is None:
            nrows = len(subjects)
    ncols =  (len(features))*2+5 
    lw=1.5 #linewidth of simulation
    cz=1.5 #capsize of data
    ms=2 #markersize of data
    y=1.2 # annotation location
    x=0.2 #annotation location
    datacol=[0.4, 0.4, 0.4]
    simcol=cols[1]
    labels_iter = iter(labels)
    pindex_in = pindex
    for sidx,subj in enumerate(subjects):
        # load UC results
        UC_data = {}
        if subjin is not None:
            s = subjin
        else:
            s = subj
        if os.path.exists(f'parameters/avatar_exercise/UC/UC_PPL_{s}.json'):
            with open(f'parameters/avatar_exercise/UC/UC_PPL_{s}.json', 'r') as file:
                UC_data = json.load(file) #contains both rest and exercise
        
        if sims is None:
            datacol=colorpicker(subj,backupcolor=simcol)
        
        if percent:
            if data is not None:
                T = data[subj]['input']['T']['mean']
            else:
                T = max(sims[subj].time_vector)
        else:
            T =  1

        # all wanted features for the heartbeats
        if allinone:
            pindex=pindex_in
        for idx, feat in enumerate(features):
            pindex +=1
            ax=plt.subplot(nrows,ncols,(pindex,pindex+1))
            pindex +=1
            if data is not None and feat in data[subj]:
                plotted = plot_datafeature(data[subj][feat],feat,c=datacol,ms=ms,cz=cz,T=T)
            if sims is not None:
                if simsSS is None:
                    ss = None
                else:
                    ss = simsSS[subj]
                if params is None:
                    ps = None
                    model = None
                else:
                    ps = params[subj]
                    model = 'avatar'
            if sims is not None and sims[subj] is not None:
                plot_sim(ps, sims[subj], [],c=colorpicker(subj,backupcolor=simcol), feature_to_plot=feat,model=model,lw=lw,simSS=ss,T=T)
                if UC_data and subj in UC_data:
                    ucvals = UC_data[subj]
                    if feat in ucvals and ucvals[feat]['Min'] is not None and ucvals[feat]['Max'][0] is not None:
                        ax.fill_between(np.divide(np.array(ucvals['Time']),T), 
                                        ucvals[feat]['Min'], 
                                        ucvals[feat]['Max'], 
                                        alpha=0.4, color=colorpicker(subj,backupcolor=simcol))
            if percent:
                plt.xlim(-0.05,1.05)
                plt.xticks([0,1],[0,100])
            elif sims is not None:
                plt.xlim(-0.05,sims[subj].time_vector[-1])
                plt.xticks([0,sims[subj].time_vector[-1]],[0,round(sims[subj].time_vector[-1],1)])
            else:
                plt.xlim(-0.05,data[subj]['input']['T']['mean'])
                plt.xticks([0,data[subj]['input']['T']['mean']],[0,round(data[subj]['input']['T']['mean'],1)])

            if idx == 0:
                if dofigure and not allinone:
                    ax.annotate(next(labels_iter,'')+') '+subj,xy=(-0.7,y),xycoords='axes fraction',fontsize='large',color='k',va='top', ha='left')
                else:
                    plt.ylabel(figname)
            if dofigure:
                ax.annotate(feat,xy=(x,y),xycoords='axes fraction',fontsize='large',color='k',va='top', ha='center')
                if sidx == len(subjects)-1:
                    if percent:
                        plt.xlabel(f'% of cardiac cycle')
                    else:
                        plt.xlabel('Time (s)')
                    

        if allinone and sims is not None:
            x=list(sims.keys()).index(subj)*0.25
        else:
            x=0
        # blood pressure
        pindex += 1 
        ax=plt.subplot(nrows,ncols,pindex)
        if data is not None and 'SBP' in data[subj]:
            dat = copy.deepcopy(data[subj]['SBP'])
            dat['time'] = 0+x
            plot_datafeature(dat,'SBP',c=datacol,marker='o',ms=ms,cz=cz)
            dat = copy.deepcopy(data[subj]['DBP'])
            dat['time'] = 1+x
            plot_datafeature(dat,'DBP',c=datacol,marker='o',ms=ms,cz=cz)

        if sims is not None and sims[subj] is not None and sims[subj].feature_data is not None and 'Pao' in sims[subj].feature_names:
            pao=sims[subj].feature_data[:,sims[subj].feature_names.index('Pao')]
            if  data is not None and 'SBP' in data[subj]:
                plt.plot(0+x,estimate_sbp(sims[subj],data[subj]['SBP']['mean']),c=colorpicker(subj,backupcolor=simcol),marker='o',fillstyle='none',ms=ms*1.5)
            else:
                plt.plot(0+x,max(pao),c=colorpicker(subj,backupcolor=simcol),marker='o',fillstyle='none',ms=ms*1.5)
            plt.plot(1+x,min(pao),c=colorpicker(subj,backupcolor=simcol),marker='o',fillstyle='none',ms=ms*1.5)

            if UC_data and subj in UC_data:
                ucvals = UC_data[subj]
                if 'SBP' in ucvals and ucvals['SBP']['Max'][0] is not None:
                    plt.plot([0+x,0+x],[ucvals['SBP']['Min'],ucvals['SBP']['Max']],c=colorpicker(subj,backupcolor=simcol),linestyle='-',lw=1,alpha=0.4)
                if 'DBP' in ucvals and ucvals['DBP']['Max'][0] is not None:
                    plt.plot([1+x,1+x],[ucvals['DBP']['Min'],ucvals['DBP']['Max']],c=colorpicker(subj,backupcolor=simcol),linestyle='-',lw=1,alpha=0.4)

        plt.xlim(-0.5,1.5+x)
        if dofigure:
            plt.xticks([0,1],['SBP','DBP'],rotation=15)
        else:
            plt.xticks([0,1],[])
        pindex+=1
        ax=plt.subplot(nrows,ncols,pindex)
        if data is not None and 'EDV' in data[subj]:
            dat = copy.deepcopy(data[subj]['EDV'])
            dat['time'] = 0+x
            plot_datafeature(dat,'EDV',c=datacol,ms=ms,cz=cz)
        if data is not None and 'ESV' in data[subj]:
            dat = copy.deepcopy(data[subj]['ESV'])
            dat['time'] = 1+x
            plot_datafeature(dat,'ESV',c=datacol,ms=ms,cz=cz)
        if sims is not None and sims[subj] is not None and 'Vlv' in sims[subj].feature_names and sims[subj].feature_data is not None:
            lvvol=sims[subj].feature_data[:,sims[subj].feature_names.index('Vlv')]
            plt.plot(0+x,max(lvvol),c=colorpicker(subj,backupcolor=simcol),marker='o',fillstyle='none',ms=ms*1.5)
            plt.plot(1+x,min(lvvol),c=colorpicker(subj,backupcolor=simcol),marker='o',fillstyle='none',ms=ms*1.5)
            
            if UC_data and subj in UC_data:
                ucvals = UC_data[subj]
                if 'EDV' in ucvals and ucvals['EDV']['Max'][0] is not None:
                    plt.plot([0+x,0+x],[ucvals['EDV']['Min'],ucvals['EDV']['Max']],c=colorpicker(subj,backupcolor=simcol),linestyle='-',lw=1,alpha=0.4)
                if 'ESV' in ucvals and ucvals['ESV']['Max'][0] is not None:
                    plt.plot([1+x,1+x],[ucvals['ESV']['Min'],ucvals['ESV']['Max']],c=colorpicker(subj,backupcolor=simcol),linestyle='-',lw=1,alpha=0.4)
        plt.xlim(-0.5,1.5+x)
        if dofigure:
            plt.xticks([0,1],['EDV','ESV'],rotation=15)
        else:
            plt.xticks([0,1],[])

        # parameters
        paramnames = {'Emax_LV','Caa','ELCo'}
        for d,p in enumerate(paramnames):
            pindex+=1
            ax=plt.subplot(nrows,ncols,pindex)
            if data is not None and p in data[subj]:
                dat = copy.deepcopy(data[subj][p])
                dat['time'] = 0+x
                plot_datafeature(dat,p,c=datacol,ms=ms,cz=cz)
            if sims is not None and sims[subj] is not None and p in sims[subj].feature_names and sims[subj].feature_data is not None:
                if p == 'Emax_LV':
                    val=sims[subj].feature_data[:,sims[subj].feature_names.index('Elv')]
                else:
                    val=sims[subj].feature_data[:,sims[subj].feature_names.index(p)]
                plt.plot(0+x,max(val),c=colorpicker(subj,backupcolor=simcol),marker='o',fillstyle='none',ms=ms*1.5)
                            
                if UC_data and subj in UC_data:
                    ucvals = UC_data[subj]
                    if p in ucvals and ucvals[p]['Max'] is not None and ucvals[p]['Max'][0] is not None:
                        plt.plot([0+x,0+x],[ucvals[p]['Min'],ucvals[p]['Max']],c=colorpicker(subj,backupcolor=simcol),linestyle='-',lw=1,alpha=0.4)
            plt.xlim(-0.5,0.5+x)
            if not allinone and  sims is not None:
                plt.ylim(0,max(val)*2)
            if dofigure:
                plt.xticks([0],[p])
            else:
                plt.xticks([0],[])
    if savefigs and dofigure:
        fig.savefig(f"./figures/{figname}.png", dpi=300)
        fig.savefig(f"./figures/{figname}.svg")
    if not dofigure:
        return pindex

def plot_allinone_avatar(sims=None,simsSS=None,data=None,figname='All_subjects_avatarexercise',
                         savefigs=True,allinone=False,percent=False,subjects=None,anonymos=True,nrows=None):
    if subjects is None:
        if sims is not None:
            subjects = list(sims.keys())
        else:
            subjects = list(data.keys())
    if nrows is None:
        nrows = int(len(subjects))

    cm = 1/2.54  # centimeters in inches
    fig=plt.figure(figname,figsize=(20*cm, 29*cm),layout="tight")
    plt.rcParams.update({'font.family': 'serif','font.size': 7})
    pindex=0
    letters = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
    for i, subj in enumerate(subjects):
        if anonymos:
            sname = subj[0]+letters[i]
        else:
            sname = subj
        # no params --> no new sim is done, only plotting
        if sims is not None:
            pindex=plot_avatar_all(params=None,data=data[subj],sims=sims[subj],simsSS=simsSS[subj],savefigs=False,
                                   figname=sname,dofigure=False,nrows=nrows,pindex=pindex,allinone=allinone,percent=percent)
        else:
            pindex=plot_avatar_all(params=None,data=data[subj],sims=None,simsSS=None,savefigs=False,
                                   figname=sname,dofigure=False,nrows=nrows,pindex=pindex,allinone=allinone,percent=percent) # assuming that simulations are already done
    if savefigs:
        fig.savefig(f"./figures/{figname}.png", dpi=300)
        fig.savefig(f"./figures/{figname}.svg")

def plot_severalfeatures(params,generalmodel,modelname,sim, data, features_to_plot=[['SV', 'EDV', 'ESV']],
                         ylabels=['LV volumes (mL)'],simSS=None,savefig=False,figname='SV_EDV_ESV',
                         wide=False,separate_cols=False,allexpsinone=False,testparams=None,percent=False,ncols=None,figwidth=None,lw=1,subj=None):
    # Set up the figure
    if allexpsinone:
        if ncols is None:
            ncols=4
        nrows = int(np.ceil(len(features_to_plot)/ncols))
    else:
        if ncols is None:
            ncols=len(data)
        nrows = len(features_to_plot) 
        
    cm = 1/2.54  # centimeters in inches
    if figwidth is None:
        if ncols > 5:
            figwidth = ncols*2*cm
        else:
            figwidth =ncols*3.35*cm
    else:
        figwidth = figwidth*cm

    if wide:
        rowheight = 2.8
    else:
        rowheight = 4

    fig = plt.figure(figname,figsize=(figwidth, rowheight*nrows*cm),layout="constrained")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
    plotnum = 1   

    # load UC results
    UC_data = {}
    if generalmodel == "avatar_exercise" and subj is not None:
        if os.path.exists(f'parameters/avatar_exercise/UC/UC_PPL_{subj}.json'):
            with open(f'parameters/avatar_exercise/UC/UC_PPL_{subj}.json', 'r') as file:
                UC_data = json.load(file)
    else:
        if os.path.exists(f'parameters/{modelname}/UC/UC_PPL_{modelname}.json'):
            with open(f'parameters/{modelname}/UC/UC_PPL_{modelname}.json', 'r') as file:
                UC_data = json.load(file)

    # loop through the features
    if generalmodel == 'avatar_exercise':
        simmodel = 'avatar'
    else:
        simmodel = generalmodel
    linestyles = ['-','--',':','.','-.']*int(len(data)/5+1)
    for i, features in enumerate(features_to_plot):
        if len(features)<2:
            symbols = ['.']
        else:
            symbols = ['x', '^', 's','*','o','<','>']  # Different markers for SV, EDV, ESV
        ylabel = ylabels[i]
        # Plot the features for each experiment
        if allexpsinone:
            ax=plt.subplot(nrows,ncols,plotnum)
            plotnum += 1
        for d,exp in enumerate(data):
            if not allexpsinone:
                if d == 0:
                    ax = plt.subplot(nrows,ncols,plotnum)
                    axone=ax
                    plt.grid(axis='y')
                else:
                    ax=plt.subplot(nrows,ncols,plotnum,sharey=axone)
                    ax.tick_params(left=False,labelleft=False)  # Hide y-tick labels only
                    plt.grid(axis='y')
                plotnum += 1

            if simSS is not None:
                ss = simSS[exp]
            else:
                ss = None

            colors = plt.cm.Set2(np.linspace(0, 1, len(features)))
            isvariables=False
            for feature, symbol,c in zip(features, symbols,colors):
                if separate_cols:
                    col = c
                else:
                    col = colorpicker(exp, backupcolor=[0.2,0.2,0.2],model=generalmodel)
                if feature in data[exp]  and generalmodel != 'avatar' and simmodel !='avatar':
                    if percent:
                        T=data[exp]['input']['T']['mean']
                    else:
                        T=1
                    if allexpsinone:
                        plot_datafeature(data[exp][feature], feature, c=np.multiply(mplt.colors.to_rgb(col),0.5), marker='.', lw=0.5*lw, cz=1.5, ms=5,T=T)
                    else:
                        plot_datafeature(data[exp][feature], feature, c=[0.1,0.1,0.1], marker=symbol, lw=0.5*lw, cz=3, ms=5,T=T)
                varnames =  ['k_diast_LV','k_syst_LV','onset_LA','m2_LV','Caa','Rleg_ex','Rupper','Rlower','Rbody','Rtot']
                if exp in sim and (feature in sim[exp].feature_names or feature in ['PV_LV','PV_LA']):
                    simtime = np.arange(data[exp]['input']['simulationtime'][0],data[exp]['input']['simulationtime'][-1],data[exp]['input']['temporal_res'])
                    if isinstance(params, dict):
                        p = params[exp]
                    else:
                        p = params
                    if percent:
                        T=simtime[-1]
                    else:
                        T=1

                    if generalmodel == 'avatar_exercise' and feature in varnames and sim[exp].feature_data is not None:
                        x=features.index(feature)
                        fidx=sim[exp].feature_names.index(feature)
                        plt.plot(x,np.log(sim[exp].feature_data[-1,fidx]),marker='o',c=col,ms=5,lw=0)
                        isvariables=True
                    else:
                        if allexpsinone:
                            ls = linestyles[d]
                        else:
                            ls='-'
                        if feature in ['PV_LV','PV_LA']: 
                            xfeat='Vlv' if feature == 'PV_LV' else 'Vla'
                            yfeat = 'P_LV' if feature == 'PV_LV' else 'P_LA'
                            plot_sim(p, sim[exp], simtime, secondfeature=yfeat,feature_to_plot = xfeat,model=simmodel, simSS=ss, lw=lw, ls=ls, c=col,T=T)
                            mvopen=sim[exp].feature_data[:,sim[exp].feature_names.index('Mitral valve open')]
                            mask=mvopen[10:]>0 # make sure not the first time point is included if shifted flow
                            first_open = np.argmax(mask) + 10 if np.any(mask) else None
                            last_open = len(mask) - 1 - np.argmax(mask[::-1]) if np.any(mask) else None
                            last_open = min(len(mask)-1,last_open+1)
                            if first_open is not None:
                                plt.plot(sim[exp].feature_data[first_open,sim[exp].feature_names.index(xfeat)],sim[exp].feature_data[first_open,sim[exp].feature_names.index(yfeat)],'o',c=col,ms=7,fillstyle='none',lw=2)
                            if last_open is not None:
                                plt.plot(sim[exp].feature_data[last_open,sim[exp].feature_names.index(xfeat)],sim[exp].feature_data[last_open,sim[exp].feature_names.index(yfeat)],'o',c=col,ms=6)
                            if feature == 'PV_LV':
                                avopen=sim[exp].feature_data[:,sim[exp].feature_names.index('Aortic valve open')]
                                mask=avopen[:-int(len(avopen)*0.3)]>0 # make sure not the last time point is included if shifted flow
                                first_open = np.argmax(mask) if np.any(mask) else None
                                last_open = len(mask) - 1 - np.argmax(mask[::-1]) if np.any(mask) else None
                                last_open = min(len(mask)-1,last_open+1)
                                if first_open is not None:
                                    plt.plot(sim[exp].feature_data[first_open,sim[exp].feature_names.index(xfeat)],sim[exp].feature_data[first_open,sim[exp].feature_names.index(yfeat)],'^',c=col,ms=7,fillstyle='none',lw=2)
                                if last_open is not None:
                                    plt.plot(sim[exp].feature_data[last_open,sim[exp].feature_names.index(xfeat)],sim[exp].feature_data[last_open,sim[exp].feature_names.index(yfeat)],'^',c=col,ms=6)
                        else:
                            plot_sim(p, sim[exp], simtime, feature_to_plot=feature, model=simmodel, simSS=ss, lw=lw, ls=ls, label=feature, c=col,T=T)
                            if testparams is not None:
                                colors = ['c','r','g','b']
                                for colidx,param in enumerate(testparams):
                                    for pidx, value in enumerate(testparams[param]):
                                        testp = copy.deepcopy(p)
                                        testp[sim[exp].parameter_names.index(param)] = value
                                        plot_sim(testp, sim[exp], simtime, feature_to_plot=feature, model=simmodel, simSS=ss, lw=lw, ls='--', label=f'{feature} {param}={value}', c = np.multiply(mplt.colors.to_rgb(colors[colidx]), pidx/len(testparams[param])),T=T)

                if UC_data and exp in UC_data and feature in UC_data[exp]:
                    if feature in ['PV_LA','PV_LV'] and UC_data[exp][feature]['Min_x'][0] is not None and UC_data[exp][feature]['Max_x'][0] is not None:
                        pass
                    elif  UC_data[exp][feature]['Min'] is not None and UC_data[exp][feature]['Min'][0] is not None and UC_data[exp][feature]['Max'][0] is not None:
                        ax.fill_between(np.array(UC_data[exp]['Time'])/T, 
                                        UC_data[exp][feature]['Min'], 
                                        UC_data[exp][feature]['Max'], 
                                        alpha=0.4, color=col)
            if separate_cols and d==0:
                plt.legend(fontsize=6)
                    
            if 'Workload' in sim[exp].feature_names:
                wattsim = sim[exp].feature_data[:,sim[exp].feature_names.index('Workload')]
                plot_exercise_times(ax, wattsim, np.array(sim[exp].time_vector)/T,fc='#eb9e26ff')

            if feature in ['PV_LA','PV_LV']:
                left, right = plt.xlim()
                bottom, top = plt.ylim()
                xvals = [left,left + ((right-left)/2),right]
                plt.xticks([int(x) for x in xvals])
                yvals = [0,(top-bottom)/2,top]
                plt.yticks([int(y) for y in yvals])

            if feature == 'PV_LA':
                plt.ylabel('LA pressure (mmHg)')
                plt.xlabel('LA volume (mL)')
            elif feature == 'PV_LV':
                plt.ylabel('LV pressure (mmHg)')
                plt.xlabel('LV volume (mL)')
            elif exp == list(data.keys())[0]:
                plt.ylabel(f'{ylabel}')
                
            if percent:
                plt.xlabel(f'% of cardiac cycle')
                plt.xlim([0,100])
            elif isvariables:
                plt.xticks(range(len(features)), features)
            elif feature not in ['PV_LV','PV_LA']:
                if allexpsinone or i == len(features_to_plot)-1:
                    plt.xlabel(f'Time ({sim[exp].time_unit})')
            if i == 0:
                plt.title(f'{exp}')

    if savefig:
        fig.savefig(f"./figures/{figname}.png", dpi=300)
        fig.savefig(f"./figures/{figname}.svg")

def plot_data_response(data,features_to_plot,figname='data response',lw=1,ms=3,savefig=False,averagedata=None):
    # Set up the figure
    ncols=2
    nrows = len(features_to_plot)
    cm = 1/2.54  # centimeters in inches
    figwidth = 9.5*cm
    fig = plt.figure(figname,figsize=(figwidth, 4.5*nrows*cm),layout="constrained")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 7})
    plotnum = 1   
    # loop through the features
    for feature in features_to_plot:
        ax=plt.subplot(nrows,ncols,plotnum)
        plotnum += 1
        for exp in data:
            if feature in data[exp] or feature == 'COi' and 'HR' in data[exp] and 'SVi' in data[exp]:
                if feature == 'COi' or 'mean' in data[exp][feature] and data[exp][feature]['mean'] is not None:
                    if feature == 'COi':
                        HR = np.array(data[exp]['HR']['mean'])
                        SVtime = np.array(data[exp]['SVi']['time'])
                        HRtime = np.array(data[exp]['HR']['time'])
                        HR = np.interp(SVtime, HRtime, HR)  # Interpolate HR to SV time points
                        meanval = HR * np.array(data[exp]['SVi']['mean']) / 1000  # Convert to L/min
                        time = np.array(data[exp]['SVi']['time'])
                    else:
                        meanval = np.array(data[exp][feature]['mean'])
                        time = np.array(data[exp][feature]['time'])
                    if exp[0] != 'c' and exp[0] != 'p':
                        rest = meanval[time <= data[exp]['input']['activitystarttime']][0] if len(meanval[time <= data[exp]['input']['activitystarttime']]) > 0 else None
                        ex = meanval[time > data[exp]['input']['activitystarttime']][0] if len(meanval[time > data[exp]['input']['activitystarttime']]) > 0 else None
                        ramp = None
                        rec = None
                        xvals = [0, 1]
                        response = [rest, ex]
                    else:
                        inputtime = np.array(data[exp]['input']['time'])
                        rest = meanval[time <= inputtime[1]][0] if len(meanval[time <= inputtime[1]]) > 0 else None
                        ex = meanval[(time > inputtime[1]) & (time < inputtime[2])][0] if len(meanval[(time > inputtime[1]) & (time < inputtime[2])]) > 0 else None
                        ramp = meanval[(time > inputtime[2]) & (time < inputtime[3])][0] if len(meanval[(time > inputtime[2]) & (time < inputtime[3])]) > 0 else None
                        rec = meanval[(time > inputtime[3]) & (time < inputtime[4])][0] if len(meanval[(time > inputtime[3]) & (time < inputtime[4])]) > 0 else None
                        response = [rest, ex, ramp, rec]
                        xvals = [0, 1, 2, 3]
                    xvals = [val for i,val in enumerate(xvals) if response[i] is not None]
                    response = [val for val in response if val is not None]
                    response = np.array(response)
                    if rest is not None:
                        if exp[0] != 'c' and exp[0] != 'p':
                            marker = '*'
                        else:
                            marker = 'o'
                        plt.plot(xvals,100*(response-rest)/rest,ls='-',marker=marker,c=colorpicker(exp),ms=ms,lw=lw,alpha=0.4)          
        if averagedata is not None:
            for exp, dat in averagedata.items():
                if feature+'perc' not in dat:
                    if feature[0:3] == 'COi' and 'SVi' in dat and 'HR' in dat:
                        time = np.array(dat['HR']['time'])
                        meanval = np.array(np.divide(np.multiply(dat['SVi']['mean'],dat['HR']['mean']),1000))
                    else:
                        time = np.array(dat[feature]['time']) if feature in dat else None
                        meanval = np.array(dat[feature]['mean']) if feature in dat else None
                    if meanval is not None:
                        rest = meanval[time <= dat['input']['activitystarttime']][0] if len(meanval[time <= dat['input']['activitystarttime']]) > 0 else None
                        ex = meanval[time > dat['input']['activitystarttime']][0] if len(meanval[time > dat['input']['activitystarttime']]) > 0 else None
                        xvals = [0, 1]
                        response = [0, np.divide(ex-rest,rest)] if rest is not None else [0, None]
                        svals = [0,0]
                    else:
                        response = None
                else:
                    response = dat[feature+'perc']['mean'] if feature+'perc' in dat else None
                    svals = dat[feature+'perc']['SEM'] if feature+'perc' in dat else None
                    if len(response) != 4:
                        xvals = []
                        for t in dat[feature+'perc']['time']:
                            if t <= dat['input']['time'][1]:
                                xvals.append(0)
                            elif t <= dat['input']['time'][2]:
                                xvals.append(1)
                            elif t <= dat['input']['time'][3]:
                                xvals.append(2)
                            else:
                                xvals.append(3)
                    else:
                        xvals = [0, 1, 2, 3]
                if response is not None:
                    xvals = [val for i,val in enumerate(xvals) if response[i] is not None and response[i] is not np.nan]
                    yvals = [val for i,val in enumerate(response) if val is not None and val is not np.nan]
                    svals = [val for i,val in enumerate(svals) if val is not None and val is not np.nan]
                    if exp == 'average_pat_nop13':
                        c=colorpicker('average_pat')
                        ls='--'
                    else:
                        c=colorpicker(exp)
                        ls='-'
                    plt.errorbar(xvals, np.multiply(100,yvals), np.multiply(100,svals), marker='^',ls=ls, label=exp, c=c, ms=ms*2, lw=lw*2,capsize=ms*2)

            # statistical test
            if 'average_pat' in averagedata and 'average_control' in averagedata: 
                from scipy.stats import ttest_rel
                cont = averagedata['average_control'][feature+'perc']['alldata']
                pat = averagedata['average_pat'][feature+'perc']['alldata']
                bottom, top = ax.get_ylim()
                for i, key in enumerate(cont):
                    if i ==0 or len(cont[key])<1 or len(pat[key])<1:
                        continue
                    #parametric paired test
                    nanvals = np.isnan(cont[key]) | np.isnan(pat[key])
                    t_stat, p_val = ttest_rel(cont[key][~nanvals], pat[key][~nanvals])
                    #NOTE: paired ttest does not work on the anonymized data since the pairing is removed, so the p-values will be incorrect. 
                    # Use the original data for correct p-values.
                    # The p-values are therefore not plotted in this public verison of the code.
                    '''
                    if p_val < 0.001:
                        plt.text(i*0.8, top - abs(top)*0.2, f'p <\n0.001*', color='black', fontsize=6)
                    elif p_val < 0.05:
                        plt.text(i*0.8, top - abs(top)*0.2, f'p =\n{p_val:.3f}*', color='black', fontsize=6)
                    elif not np.isnan(p_val) and p_val is not None:
                        plt.text(i*0.8, top - abs(top)*0.2, f'p =\n{p_val:.3f}', color='black', fontsize=6)
                    else:
                        plt.text(i*0.8, top - abs(top)*0.2, f'p =\nnan', color='black', fontsize=6)
                    '''

        plt.axhline(y=0, c='k', ls='--', lw=0.5)
        plt.xticks([0,1,2,3],['Rest','Level 1','Level 2','Recovery'],rotation=-30)
        plt.ylabel(f'{feature} (% from rest)')

        ax=plt.subplot(nrows,ncols,plotnum)
        plotnum += 1
        for exp in data:
            if feature in data[exp] or feature == 'COi' and 'HR' in data[exp] and 'SVi' in data[exp]:
                if feature == 'COi' or 'mean' in data[exp][feature] and data[exp][feature]['mean'] is not None:
                    if feature == 'COi':
                        HR = np.array(data[exp]['HR']['mean'])
                        SVtime = np.array(data[exp]['SVi']['time'])
                        HRtime = np.array(data[exp]['HR']['time'])
                        HR = np.interp(SVtime, HRtime, HR)  # Interpolate HR to SV time points
                        meanval = HR * np.array(data[exp]['SVi']['mean']) / 1000  # Convert to L/min
                        time = np.array(data[exp]['SVi']['time'])
                    else:
                        meanval = np.array(data[exp][feature]['mean'])
                        time = np.array(data[exp][feature]['time'])
                    if exp[0] != 'c' and exp[0] != 'p':
                        rest = meanval[time <= data[exp]['input']['activitystarttime']][0] if len(meanval[time <= data[exp]['input']['activitystarttime']]) > 0 else None
                        ex = meanval[time > data[exp]['input']['activitystarttime']][0] if len(meanval[time > data[exp]['input']['activitystarttime']]) > 0 else None
                        rec = None
                        xvals = [0, 1]
                        response = [rest, ex]
                    else:
                        inputtime = np.array(data[exp]['input']['time'])
                        rest = meanval[time <= inputtime[1]][0] if len(meanval[time <= inputtime[1]]) > 0 else None
                        ex = meanval[(time > inputtime[1]) & (time < inputtime[2])][0] if len(meanval[(time > inputtime[1]) & (time < inputtime[2])]) > 0 else None
                        ramp = meanval[(time > inputtime[2]) & (time < inputtime[3])][0] if len(meanval[(time > inputtime[2]) & (time < inputtime[3])]) > 0 else None
                        rec = meanval[(time > inputtime[3]) & (time < inputtime[4])][0] if len(meanval[(time > inputtime[3]) & (time < inputtime[4])]) > 0 else None
                        response = [rest, ex, ramp, rec]
                        xvals = [0, 1, 2, 3]
                    xvals = [val for i,val in enumerate(xvals) if response[i] is not None]
                    response = [val for val in response if val is not None]
                    response = np.array(response)
                    if rest is not None:
                        if exp[0] != 'c' and exp[0] != 'p':
                            marker = '*'
                        else:
                            marker = 'o'
                        plt.plot(xvals,response,ls='-',marker=marker,c=colorpicker(exp,rgb=True),ms=ms,lw=lw,alpha=0.4)
        if averagedata is not None:
            for exp, dat in averagedata.items():
                if exp[0:7] != 'average':
                    if feature in dat:
                        time = np.array(dat[feature]['time']) if feature in dat else None
                        meanval = np.array(dat[feature]['mean']) if feature in dat else None
                        rest = meanval[time <= dat['input']['activitystarttime']][0] if len(meanval[time <= dat['input']['activitystarttime']]) > 0 else None
                        ex = meanval[time > dat['input']['activitystarttime']][0] if len(meanval[time > dat['input']['activitystarttime']]) > 0 else None
                        rec = None
                        xvals = [0, 1]
                        response = [rest, ex]
                        svals = [0, 0]
                    elif feature == 'COi' and 'SVi' in dat and 'HR' in dat:
                        time = np.array(dat['HR']['time'])
                        meanval = np.array(np.divide(np.multiply(dat['SVi']['mean'],dat['HR']['mean']),1000))
                        rest = meanval[time <= dat['input']['activitystarttime']][0] if len(meanval[time <= dat['input']['activitystarttime']]) > 0 else None
                        ex = meanval[time > dat['input']['activitystarttime']][0] if len(meanval[time > dat['input']['activitystarttime']]) > 0 else None
                        rec = None
                        xvals = [0, 1]
                        response = [rest, ex]
                        svals = [0, 0]
                    else:
                        response = None
                else:
                    response = dat[feature]['mean'] if feature in dat else None
                    svals = dat[feature]['SEM'] if feature in dat else None
                    if len(response) != 4 and response is not None:
                        xvals = []
                        for t in dat[feature+'perc']['time']:
                            if t <= dat['input']['time'][1]:
                                xvals.append(0)
                            elif t <= dat['input']['time'][2]:
                                xvals.append(1)
                            elif t <= dat['input']['time'][3]:
                                xvals.append(2)
                            else:
                                xvals.append(3)
                    else:
                        xvals = [0, 1, 2, 3]
                if response is not None:
                    xvals = [val for i,val in enumerate(xvals) if response[i] is not None and response[i] is not np.nan]
                    yvals = [val for i,val in enumerate(response) if val is not None and val is not np.nan]
                    svals = [val for i,val in enumerate(svals) if val is not None and val is not np.nan]
                    if exp == 'average_pat_nop13':
                        c=colorpicker('average_pat')
                        ls='--'
                    else:
                        c=colorpicker(exp)
                        ls='-'
                    plt.errorbar(xvals, yvals, svals, marker='^',ls=ls, label=exp, c=c, ms=ms*2, lw=lw*2,capsize=ms*2)

            # statistical test
            if averagedata is not None and 'average_pat' in averagedata and 'average_control' in averagedata: 
                from scipy.stats import ttest_ind
                cont = averagedata['average_control'][feature]['alldata']
                pat = averagedata['average_pat'][feature]['alldata']
                bottom, top = ax.get_ylim()
                for i,key in enumerate(cont):
                    if len(cont[key])<1 or len(pat[key])<1:
                        continue
                    #t_stat, p_val = ttest_ind(cont[key], pat[key])
                    nanvals = np.isnan(cont[key]) | np.isnan(pat[key])
                    t_stat, p_val = ttest_rel(cont[key][~nanvals], pat[key][~nanvals])
                    #NOTE: paired ttest does not work on the anonymized data since the pairing is removed, so the p-values will be incorrect. 
                    # Use the original data for correct p-values.
                    # The p-values are therefore not plotted in this public verison of the code.
                    '''
                    if p_val < 0.001:
                        plt.text(i*0.8, top - abs(top)*0.1, f'p <\n0.001*', color='black', fontsize=6)
                    elif p_val < 0.05:
                        plt.text(i*0.8, top - abs(top)*0.1, f'p =\n{p_val:.3f}*', color='black', fontsize=6)
                    elif not np.isnan(p_val) and p_val is not None:
                        plt.text(i*0.8, top - abs(top)*0.1, f'p =\n{p_val:.3f}', color='black', fontsize=6)
                    '''
        plt.xticks([0,1,2,3],['Rest','Level 1','Level 2','Recovery'],rotation=-30)
        plt.ylabel(f'{feature} ()')

    if savefig:
        fig.savefig(f"./figures/{figname}.png", dpi=300)
        fig.savefig(f"./figures/{figname}.svg") 
        
def plot_data_vs_sim_corr(data,sim,features,modelname,figname='',savefig=False,plotexp=None):
    # Set up the figure
    ncols=4 #3
    nrows = int(np.ceil(len(features)/ncols))
    cm = 1/2.54  # centimeters in inches
    figwidth = 16.5*cm
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, num=figname,figsize=(figwidth, 3.5*nrows*cm),layout="constrained")
    axs = axs.flatten()
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    for i, feature in enumerate(features):
        ax = axs[i]
        for exp in data:
            if feature in data[exp]:
                y_sim = sim[exp].feature_data[:, sim[exp].feature_names.index(feature)]
                y_sim = y_sim[np.searchsorted(sim[exp].time_vector, data[exp][feature]['time'])]
                ax.scatter(data[exp][feature]['mean'], y_sim, alpha=0.5)
                ax.set_title(f'{feature}')
            elif isinstance(sim[exp], dict):
                markers = ['o', 'x', 's', '^', 'D']  # Different markers for different sexps
                for sexp in data[exp]:
                    color = colorpicker(sexp, backupcolor=[0.2,0.2,0.2],model=modelname)
                    if isinstance(sim[exp][sexp], dict) and 'input' not in data[exp][sexp]:
                        if plotexp is not None and plotexp in data[exp][sexp] and feature in data[exp][sexp][plotexp]:
                            dat = data[exp][sexp][plotexp][feature]['mean']
                            dat_t = data[exp][sexp][plotexp][feature]['time']
                        elif plotexp is not None and plotexp in data[exp][sexp] and feature in data[exp][sexp][plotexp]['input']:
                            dat = data[exp][sexp][plotexp]['input'][feature]['mean']
                            dat_t = data[exp][sexp][plotexp]['input'][feature]['time']
                        else:
                            continue
                        if dat is None:
                            continue
                    elif feature in data[exp][sexp] and data[exp][sexp][feature]['mean'] is not None:
                        dat = data[exp][sexp][feature]['mean']
                        dat_t = data[exp][sexp][feature]['time']
                    elif feature in data[exp][sexp]['input'] and data[exp][sexp]['input'][feature]['mean'] is not None:
                        dat = data[exp][sexp]['input'][feature]['mean']
                        dat_t  = data[exp][sexp]['input'][feature]['time']
                    else:
                        continue

                    if isinstance(sim[exp][sexp], dict):
                        if plotexp is not None and plotexp in sim[exp][sexp] and feature in sim[exp][sexp][plotexp].feature_names:
                            y_sim = sim[exp][sexp][plotexp].feature_data[:, sim[exp][sexp][plotexp].feature_names.index(feature)]
                            if dat_t is not None:
                                y_sim =  y_sim[np.searchsorted(sim[exp][sexp][plotexp].time_vector, dat_t)]
                        else:
                            continue
                    elif feature in sim[exp][sexp].feature_names and sim[exp][sexp].feature_data is not None:
                        y_sim = sim[exp][sexp].feature_data[:, sim[exp][sexp].feature_names.index(feature)]
                        if dat_t is not None:
                            y_sim = y_sim[np.searchsorted(sim[exp][sexp].time_vector, dat_t)]
                    else:
                        continue
                    ax.scatter(dat, y_sim, s=15, color=color, marker=markers[list(data[exp].keys()).index(sexp)%len(markers)])
                    ax.set_title(f'{feature}')
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        xmin = ax.get_xlim()[0]
        xmax = ax.get_xlim()[1]
        ax.set_xlim([min(xmin, ymin), max(xmax, ymax)])
        ax.set_ylim([min(xmin, ymin), max(xmax, ymax)])
        ax.set_xticks([round(min(xmin, ymin)), round(max(xmax, ymax))])
        ax.set_yticks([round(min(xmin, ymin)), round(max(xmax, ymax))])
        ax.plot([min(xmin, ymin), max(xmax, ymax)],[min(xmin, ymin), max(xmax, ymax)], 'k--')
        ax.set_xlabel('Data')
        ax.set_ylabel('Simulations')

    if savefig:
        fig.savefig(f"./figures/{figname}.png", dpi=300)
        fig.savefig(f"./figures/{figname}.svg")