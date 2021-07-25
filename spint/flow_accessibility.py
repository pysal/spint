# coding=utf-8
# 3. FlowAccessibility = Accessibility of flow taking existing destinations
import numpy as np
import pandas as pd
import itertools
from timeit import default_timer as timer

def generate_dummy_flows():
    nodes = ['A','B','C','D','E']
    destination_masses = [60,10,10,30,50]

    all_flow = pd.DataFrame( list(itertools.product(nodes,nodes
                                               )
                             )
                       ).rename(columns = {0:'origin_ID', 1:'destination_ID'
                                          }
                               )

    masses = {'nodes': nodes,'dest_masses': destination_masses
         }

    masses = pd.DataFrame({'nodes': nodes,'dest_masses': destination_masses
                      }, columns = ['nodes','dest_masses'
                                   ]
                     )

    all_flow['volume_in_unipartite'] = [10,10,10,10,10,
                                    10,0,0,10,10,
                                    10,0,0,10,0,
                                    10,10,10,0,10,
                                    10,10,0,10,10]

    all_flow['volume_in_bipirtate'] = [0,0,10,10,10,
                          0,0,0,10,10,
                          0,0,0,0,0,
                          0,0,0,0,0,
                          0,0,0,0,0]

    all_flow['distances'] = [0,8,2,5,5,
                          8,0,10,7,4,
                          2,10,0,6,9,
                          5,7,6,0,2,
                          5,4,9,2,0]

    all_flow = all_flow.merge(masses, how = 'left', left_on = 'destination_ID', right_on = 'nodes')

    all_flow['results_all=False'] = [500, 510, 730, 230, 190,
                                     400, 890, 750, 400, 360,
                                     150, 690, 300, 300, 360,
                                     350, 780, 670, 530, 430,
                                     230, 690, 400, 370, 400]
    
    return all_flow


#-------------------------------------------------------------------------------------------------------------



def Accessibility(flow_df, all_destinations=False):
    start = timer()
     # rename teh columns so we can call them 
    flow_df = flow_df.rename(columns = {flow_df.columns[0]:'origin_ID', 
                                            flow_df.columns[1]:'dest_ID', 
                                            flow_df.columns[2]:'dist', 
                                            flow_df.columns[3]:'weight', 
                                            flow_df.columns[4]:'dest_mass'})
    
    flow_df['dist'] = flow_df['dist'].astype(int)
    flow_df['weight'] = flow_df['weight'].astype(int)
    flow_df['dest_mass'] = flow_df['dest_mass'].astype(int)
    # create binary for weight
    flow_df['v_bin'] = 1
    flow_df.loc[flow_df['weight'].isna(),'v_bin'] = 0
    flow_df.loc[flow_df['weight'] <= 0,'v_bin'] = 0
    
    # define the base matrices
    distance = np.array(flow_df.pivot_table(values='dist', index='origin_ID', columns="dest_ID"))
    mass = np.array(flow_df.pivot_table(values='dest_mass', columns="origin_ID", index='dest_ID'))
    exists = np.array(flow_df.pivot_table(values='v_bin', index='dest_ID', columns="origin_ID"))
    
    # define the base 3d array
    nrows= len(exists)
    ones = np.ones((nrows,len(flow_df.origin_ID.unique()),len(flow_df.dest_ID.unique())))
    
    # define the identity array
    idn = np.identity(nrows) 
    idn = np.where((idn==1), 0, 1)
    idn = np.concatenate(nrows * [idn.ravel()], axis = 0).reshape(nrows,nrows,nrows).T

    # multiply the distance by mass
    ard = np.array(distance)*np.array(mass)
    
    # combine all into and calculate the output
    
    if all_destinations:
        output = np.array(idn) * (np.array(nrows * [ard]
                                              )
                                     )
        
    else:
        output = (np.concatenate(nrows * [exists], axis = 0
                                ).reshape(nrows,nrows,nrows
                                         ).T 
                 ) * np.array(idn) * (np.array(nrows * [ard]
                                              )
                                     )
    
    # get the sum and covert to series
    g = pd.DataFrame((
        np.sum(output,axis = 1
                           )  ).reshape(1,len(flow_df)
                                                    ).T
                    )
    
    end = timer()
    print('time elapsed: ' + str(end - start))
    return g[0]

#-------------------------------------------------------------------------------------------------------------


def test(function):
    
    # generate data
    flow = generate_dummy_flows()
    
    # get the right columns
    flow = flow.loc[:,['origin_ID', 'destination_ID','distances', 'volume_in_unipartite','dest_masses','results_all=False']]
    
    # apply accessibility to all data
    flow['acc_uni'] = function(flow_df = flow, all_destinations=False)
    
    # check the results
    if (flow['results_all=False'] == flow['acc_uni']).all() == True:
        print('All good to go!')
    else: 
        print('Ay caramba, something is not working correctly.')