# coding=utf-8

import numpy as np
import pandas as pd
import itertools

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
                          5,7,6,0,9,
                          5,4,9,2,0]

    all_flow = all_flow.merge(masses, how = 'left', left_on = 'destination_ID', right_on = 'nodes')

    all_flow['results_all=False'] = [500, 510, 730, 580, 190,
                                     400, 890, 750, 750, 360, 
                                     150, 690, 300, 300, 360,
                                     350, 780, 670, 880, 430,
                                     230, 690, 400, 370, 400]
    
    return all_flow