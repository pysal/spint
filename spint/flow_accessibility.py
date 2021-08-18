# coding=utf-8
# 3. FlowAccessibility = Accessibility of flow taking existing destinations
import numpy as np
import pandas as pd
import itertools
from timeit import default_timer as timer

def _generate_dummy_flows():
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



def Accessibility(origins, destinations, distances, weights, masses, all_destinations=False):
    
    # convert numbers to integers
    distances = np.array(distances.astype(int))
    weights = np.array(weights.astype(int))
    masses = np.array(masses.astype(int))
    origins = np.array(origins)
    
    # define error
    if len(distances) != len(weights) != len(masses) != len(origins):
        raise ValueError("One of the input array is different length then the others, but they should all be the same length. See notebook example if you are unsure what the input should look like ")
    
    # define number of rows
    nrows= len(origins)
    uniques = len(np.unique(np.array(origins)))
    
    # create binary for weight
    v_bin =  np.ones(nrows)
    weights[np.isnan(weights)] = 0
    v_bin[weights <= 0] = 0
    
    # define the base matrices
    distance = distances.reshape(uniques,uniques)
    mass =masses.reshape(uniques,uniques).T
    exists = v_bin.reshape(uniques,uniques)
    
      
    # define the identity array
    idn = np.identity(uniques) 
    idn = np.where((idn==1), 0, 1)
    idn = np.concatenate(uniques * [idn.ravel()], axis = 0
                        ).reshape(uniques,uniques,uniques
                                 ).T

    # multiply the distance by mass
    dm = distance * mass
    
    # combine all matrices for either all or existing destinations
    if all_destinations:
        output = idn * (nrows * [dm])
        
    else:
        output = (np.concatenate(uniques * [exists], axis = 0
                                ).reshape(uniques,uniques,uniques
                                         ).T
                 ) * idn * (uniques * [dm]
                           )
    
    # get the sum and covert to series
    output = (np.sum(output,axis = 1
                    ) 
             ).reshape(nrows
                      ).T
    
    
    return output
