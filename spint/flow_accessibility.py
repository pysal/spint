# coding=utf-8
# 3. FlowAccessibility = Accessibility of flow taking existing destinations

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
    x1 = pd.DataFrame({'D': np.array([D]*len(all_dest), dtype=object), 
                       'dests':all_dest}).merge(flow_df, how='left', left_on=['D','dests'], right_on=['origin_ID','dest_ID'])
    
    # merge with the distances and masses 
    
    # Delete the flow to origin
    x1 = x1[~x1.dests.isin(list(O))]    

    # calculate the accessibility
    A = (x1['dist']*x1['dest_mass']).sum()

    return A