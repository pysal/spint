"""
Tests for Accessibility function.

The correctness of the function is veryfied by matching the output to manually calculated values, using very simple dummy dataset.

"""

__author__ = 'Lenka Hasova haska.lenka@gmail.com'

import unittest
import numpy as np
from ..flow_accessibility import _generate_dummy_flows
from ..flow_accessibility import Accessibility



class AccessibilityTest(unittest.TestCase):
    
    def test_accessibility(self):
        flow = _generate_dummy_flows()
        flow = flow.loc[:,['origin_ID', 'destination_ID','distances', 'volume_in_unipartite','dest_masses','results_all=False']]
        flow['acc_uni'] = Accessibility(nodes = flow['origin_ID'],  distances = flow['distances'], weights = flow['volume_in_unipartite'], masses = flow['dest_masses'], all_destinations=False)
        
        np.testing.testing.assert_array_equal(flow['results_all=False'], flow['acc_uni'])

if __name__ == '__main__':
    unittest.main()