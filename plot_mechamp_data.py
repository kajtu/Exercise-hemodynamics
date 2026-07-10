import matplotlib.pyplot as plt
from functions.data_plot_functions import plot_data_response
from functions.utils import get_tabdata
import json
import numpy as np

# Script for plotting data from the MeChaMP study. The script generates tables and figures for the manuscript.

savefig=True

# Load data
with open("./data/MeChaMPdata_anonymized.json", "r") as f:
    data = json.load(f)
controls = data['controls']
patients_pre = data['patients_pre']
mechampdata = {**controls, **patients_pre}  # combine patients and controls
exercisedata=mechampdata

with open("data/MeChaMPdata_average_anonymized.json", "r") as f:
    average_data = json.load(f) # These were originally calculated with the function calculate_average in functions/datafunctions.py
average_pat = average_data['average_pat']
average_control = average_data['average_control']
for feature in average_pat.keys():
    if 'alldata' in average_pat[feature].keys() and 'alldata' in average_control[feature].keys():
        p = average_pat[feature]['alldata']
        c = average_control[feature]['alldata']
        for key in p.keys():
            p[key] = [val if not val is None else np.nan for val in p[key]]
            p[key] = np.array(p[key])
            c[key] = [val if not val is None else np.nan for val in c[key]]
            c[key] = np.array(c[key])

#%% Manuscript figure 3: Measured hemodynamic respnse to rest, two levels of physical activity, and recovery in patients and controls
#NOTE: P-values are not plotted in this anonymized and public version of the code since the pairing between controls and patients is removed, so the p-values will be incorrect. 

fulldata = {exp:dat['Full'] for exp,dat in exercisedata.items()}
fulldata_pats_pre = {exp:dat['Full'] for exp,dat in patients_pre.items()}
fulldata_contr = {exp:dat['Full'] for exp,dat in controls.items()} 
plot_data_response(fulldata,['SVi','HR','COi','SBP','DBP','SVR'],figname='Figure3_data hemodynamic exercise response',
                   savefig=True,averagedata=average_data)
plt.show()


#%% Manuscript tables 4 and 5: subject characteristics and baseline hemodynamics 
# NOTE individual characteristics data are excluded for anonymization reasons. 
varstocomp = ('age','sex','weight','length','waist_circumference','BMI','BSA',
                    'sbp_office','dbp_office','SBP_home','DBP_home','HR_rest',
                    'SV_rest','ESV_rest','EDV_rest',
                     'glucose','hba1c',
                     'tg','cholesterol','non_hdl_cholesterol','ldl','hdl',
                     'hb','egfr')
#tabdata,stattable,name_unit_dict = get_tabdata(mechampdata,savetables=savefig,varstocompare=varstocomp)

