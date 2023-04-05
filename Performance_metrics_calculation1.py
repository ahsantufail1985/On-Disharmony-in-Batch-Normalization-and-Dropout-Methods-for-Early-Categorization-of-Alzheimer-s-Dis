# -*- coding: utf-8 -*-
"""

@author: AHSAN
"""


from pycm import ConfusionMatrix



cm = ConfusionMatrix(matrix={"AD": {"AD":2, "MCI":1, "NC":1}, 
                        "MCI": {"AD":3, "MCI":0, "NC":4}, 
                        "NC": {"AD":0, "MCI":2, "NC":10}})


# cm = ConfusionMatrix(matrix={"AD": {"AD":8, "NC":0}, 
#                         "NC": {"AD":1, "NC":7}}) 

print("RCI = ", cm.RCI) 
print("CEN = ", cm.CEN) 
print("IBA = ", cm.IBA)
print("Geometric Mean = ", cm.GM)
print("MCC = ", cm.MCC) 
