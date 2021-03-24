# coding=utf-8
# 3. AFED = Accessibility of flow taking existing destinations

import numpy as np
import pandas as pd

def AFED(flow_df, row_index): # AFAPF
    
      
    # We assume that 
        # the user has created dataset that has all the possible combinations of origins and destinations, to which he merged distances, flow volumes and destination mass
            # 1st column = Origin ID
            # 2nd column = Destination ID
            # 3rd column = Distance 
            # 4th column = Weight on flow / flow volume
            # 5th column = Mass on Destinations

            
     # rename the columns from an input dataset so we can call them later
    flow_df = flow_df.rename(columns = {flow_df.columns[0]:'origin_ID', 
                                            flow_df.columns[1]:'dest_ID', 
                                            flow_df.columns[2]:'dist', 
                                            flow_df.columns[3]:'weight', 
                                            flow_df.columns[4]:'dest_mass'})
    
    # define Origin and Destination for each row 
    D = flow_df['dest_ID'][row_index]
    O = flow_df['origin_ID'][row_index]
    
    # get the list of possible destinations
    all_dest = (flow_df.query('origin_ID == @O')
                .query('weight > 0')
                ['dest_ID']
                .unique()
               )    
    
    # Create all destination flows 
    x1 = pd.DataFrame({'D': np.array([D]*len(all_dest), dtype=object), 
                       'dests':all_dest}).merge(flow_df, how='left', left_on=['D','dests'], right_on=['origin_ID','dest_ID'])
    
    # merge with the distances and masses 
    
    # calculate the accessibility
    A = (x1['weight']*x1['dest_mass']).sum()

    return A