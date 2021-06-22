"""
Tests for Accessibility function.

The correctness of the function is veryfied by matching the output to manually calculated values, using very simple dummy dataset.

"""

__author__ = 'Lenka Hasova haska.lenka@gmail.com'

import unittest
import numpy as np
from ..generate_dummy_accessibility import generate_dummy_flows
from ..FlowAccessibility import Accessibility
from ..FlowAccessibility import AFED


class AccessibilityTest(unittest.TestCase):
    
    def test_accessibility(self):
        
        flow = generate_dummy_flows()
        
        flow = flow.loc[:,['origin_ID', 'destination_ID','distances', 'volume_in_unipartite','dest_masses','results_all=False']]
        
        flow['acc_uni'] = Accessibility(flow_df = flow, all_destinations=False, function = AFED)
        
        self.assertEqual(flow['results_all=False'].all(), flow['acc_uni'].all())

if __name__ == '__main__':
    unittest.main()