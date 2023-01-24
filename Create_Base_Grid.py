import os
import subprocess

def Create_Base_Grid(grid_dx,grid_dy,output_directory,FRE_tools_dir='/g/data/ul08/FRE_tools/bin/bin'):
    
    west_longitude_limit = 0
    east_longitude_limit = 360

    south_latitude_limit = -90
    north_latitude_limit = 90
    
    
    n_lon = int( (east_longitude_limit-west_longitude_limit)/grid_dx )
    n_lat = int( (north_latitude_limit-south_latitude_limit)/grid_dy ) 
    
    grid_type = 'regular_lonlat_grid'

    input_args = " --grid_type " +  grid_type 
    input_args = input_args + " --nxbnd 2 --nybnd 2" #
    input_args = input_args + f' --xbnd {west_longitude_limit},{east_longitude_limit}' #.format(yes_votes, percentage) 
    input_args = input_args + f' --ybnd {south_latitude_limit},{north_latitude_limit}'
    input_args = input_args + f' --nlon {n_lon}, --nlat {n_lat}'
    input_args = input_args + " center c_cell"

    try:
        print("MAKE HGRID",subprocess.run([fre_tools_directory + '/make_hgrid'] + input_args.split(" "),cwd = output_directory),sep = "\n")
        return 0
    except:
        print('Make_hgrid failed')
        return -9
    
if __name__ == "__main__":
    
    fre_tools_directory = '/g/data/ul08/FRE_tools/bin/bin'
    output_directory    = '/g/data/ul08/mom6_regional_input/global_grids'

    dx = 1.0/50.0
    dy = 1.0/50.0


    Create_Base_Grid(grid_dx,grid_dy=dy,output_directory=output_directory,FRE_tools_dir=fre_tools_directory)