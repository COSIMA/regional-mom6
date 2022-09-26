import numpy as np


## Written by Ashley Barnes for use with the regional scripts notebook
def sel_hgrid_indices(field,extent):
    """
    Inputs: 
        field    xarray.dataarray   the existing hgrid lon or lat to be cut down to size
        extent   list               [min,max] the lat OR lon

    Returns:
        numpy array containing the start and end indices needed to slice the hgrid/

    Function finds the indices corresponding to the start and end of some coordinate range such that the hgrid starts and ends with q points rather than t points. 
    Useful for cutting out hgrid automatically. Note that this doesn't work near the poles in the northern hemisphere.
    
    It rounds the input field so that 208.99999999 == 209, giving nice even numbers of points between whole number lat/lon bounds
    
    This function is lazily implemented. Handling the edge cases was easiest to do without vectorising, but there should be numpy functions that would make this less inefficient.
    """
    temp = False

    indices = []

    for i in range(field.shape[0]):
        if round(field[i].values.reshape(1)[0],6) >= extent[0] and temp == False:
            indices.append(2 * i)
            temp = True
        elif round(field[i].values.reshape(1)[0],6) > extent[1] and temp == True:
            indices.append((i-1) * 2 + 1) ## This goes back a step to find the last t point that was in the bounds. Then goes forward one so that the final point to slice is on the corner
            temp = False
            break
    
    return np.array(indices)

