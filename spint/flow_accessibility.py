# coding=utf-8
# 3. FlowAccessibility = Accessibility of flow taking existing destinations
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from .generate_dummy_accessibility import generate_dummy_flows

def AFED(flow_df, row_index, all_destinations=False): # AFAPF
    
    # rename teh columns so we can call them 
    flow_df = flow_df.rename(columns = {flow_df.columns[0]:'origin_ID', 
                                            flow_df.columns[1]:'dest_ID', 
                                            flow_df.columns[2]:'dist', 
                                            flow_df.columns[3]:'weight', 
                                            flow_df.columns[4]:'dest_mass'})
    # define O and D for each row the variables
    D = flow_df['dest_ID'][row_index]
    O = flow_df['origin_ID'][row_index]
    
    
    # get the list of possible destinations 
    if all_destinations:
         all_dest = (flow_df.query('origin_ID == @O')
                ['dest_ID']
                .unique()
               )
    else:
        all_dest = (flow_df.query('origin_ID == @O')
                .query('weight > 0')
                ['dest_ID']
                .unique()
               )

        
    # Create all destination flows 
    x1 = pd.DataFrame({'D': np.array([D]*len(all_dest), dtype=object
                                    ), 'dests':all_dest}
                     ).merge(flow_df, how='left', left_on=['D','dests'], right_on=['origin_ID','dest_ID'])
    
    # merge with the distances and masses 
    
    # Delete the flow to origin
    x1 = x1[~x1.dests.isin(list(O))]    

    # calculate the accessibility
    A = (x1['dist']*x1['dest_mass']).sum()

    return A

#-------------------------------------------------------------------------------------------------------------

def Accessibility(flow_df, function, all_destinations=False):
    start = timer()

    A_ij = []

    for idx in flow_df.index:
       
            if all_destinations: 
                A = function(flow_df=flow_df, row_index=idx, all_destinations=True)
            else:
                A = function(flow_df=flow_df, row_index=idx, all_destinations=False)
            A_ij.append(A)
                
    A_ij = pd.Series(A_ij)
    end = timer()

    print('time elapsed: ' + str(end - start))
    return A_ij

#-------------------------------------------------------------------------------------------------------------

def test(function):
    
    # generate data
    flow = generate_dummy_flows()
    
    # get the right columns
    flow = flow.loc[:,['origin_ID', 'destination_ID','distances', 'volume_in_unipartite','dest_masses','results_all=False']]
    
    # apply accessibility to all data
    flow['acc_uni'] = Accessibility(flow_df = flow, all_destinations=False, function = function)
    
    # check the results
    if (flow['results_all=False'] == flow['acc_uni']).all() == True:
        print('All good to go!')
    else: 
        print('Ay caramba, something is not working correctly.')