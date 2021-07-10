import numpy as np
import pandas as pd
from timeit import default_timer as timer
import itertools


def AFED_MAT(flow_df):
    
     # rename teh columns so we can call them 
    flow_df = flow_df.rename(columns = {flow_df.columns[0]:'origin_ID', 
                                            flow_df.columns[1]:'dest_ID', 
                                            flow_df.columns[2]:'dist', 
                                            flow_df.columns[3]:'weight', 
                                            flow_df.columns[4]:'dest_mass'})
    # create binary for weight
    flow_df['v_bin'] = 1
    flow.loc[flow['volume_in_unipartite'].isna(),'v_bin'] = 0
    flow.loc[flow['volume_in_unipartite'] <= 0,'v_bin'] = 0
    
    # define base matrices
    distance = flow_df.pivot_table(values='dist', index='dest_ID', columns="origin_ID")
    mass = flow_df.pivot_table(values='dest_mass', columns='dest_ID', index="origin_ID")
    exists = flow_df.pivot_table(values='v_bin', index='dest_ID', columns="origin_ID")
    
    # lists of O-D
    list_c = list(itertools.product(exists.columns,exists.index))
    df = pd.DataFrame(list_c).rename(columns = {0:'origin_ID', 1:'dest_ID'})
    
    # Start function
    array = []
    start = timer()
    
    for i,j in list_c:
        
        f = exists.loc[i,:]
        
        empty = pd.DataFrame(np.zeros((len(exists.index),len(exists.columns))), index = exists.index, columns = exists.columns)
        
        empty.loc[j,:] = f
        
        A_ij = empty *  (distance * mass)
        
        A_ij = A_ij.sum().sum()
        
        array.append(A_ij)
    
    # Get the result back to original dataframe
    df['A_ij'] = array
    flow_df.merge(df, how = 'left', on = ['origin_ID','dest_ID'])

    #how long did this took?
    end = timer()
    print('time elapsed: ' + str(end - start))
    return(flow_df)

