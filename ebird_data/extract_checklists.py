import pandas as pd
import os
import numpy as np

col_name_sp = "SCIENTIFIC NAME"
col_name_loc = "LOCALITY ID"
event_col = 'SAMPLING EVENT IDENTIFIER'
col_obs = "OBSERVATION DATE"
species = pd.read_csv("species_list.csv")
ebird_codes = list(species['ebird_code'])

dict_sp = {}
dict_sp_to_code = {}
for i in range(len(species)):
    name = species.iloc[i]['scientific_name']
    ebird_code = species.iloc[i]['ebird_code']
    dict_sp[name] = i

sightings = pd.read_csv("ebird_data.csv", sep='\t')
groups_sightings = sightings.groupby(event_col)
counter_group = 0
for group_idx, group_s in groups_sightings:
    print(f"{counter_group}/{len(groups_sightings)}")
    if counter_group >= 0:
        enc_vect = np.zeros(670)
        group_s = group_s.reset_index(drop=True)
        for i in range(len(group_s)):
            try:
                row = group_s.iloc[i]
                specie = row[col_name_sp]
                enc_vect[dict_sp[specie]] = 1
                obs = row[col_obs]
                locality = row[col_name_loc]
            except:
                pass
        path_locality = f"checklists/{locality}"
        os.makedirs(path_locality, exist_ok=True)
        dict_obs = {"ebird_code":ebird_codes, "is_observed":enc_vect}
        df_s = pd.DataFrame.from_dict(dict_obs)
        df_s['date'] = obs
        df_s.to_csv(f"{path_locality}/{group_idx}.csv", index=False)
    counter_group += 1