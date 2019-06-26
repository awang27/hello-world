from __future__ import print_function
import argparse
import json
import yaml
import os

import msgpack as mp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import matplotlib.pyplot as plt
import pprint
import random
import datetime


from dexai.unpack_scoops import unpack_scoop
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.rigid_body_tree import RigidBodyTree, RigidBodyFrame

#######################################################################################################
############# THIS SECTION HAS FUNCTIONS SPECIFYING PATHNAMES TO OBTAIN TRAJECTORIES ##################
#######################################################################################################

def get_utensil_container_dirs(workstation_name='greentown_candy', filter_utensil=None):
    '''
        Returns a list of paths specifying directories for the trajectories of the input workstation and utensil.
        If there is no specified utensil, the function will return trajectory directories corresponding to all
        available utensils for the given workstation.
    '''
    home = os.path.expanduser('~')
    traj_lib_dir = os.path.join(home, 'traj_lib')
    if filter_utensil is None:
        utensil_dirs = os.listdir(os.path.join(traj_lib_dir, workstation_name))
    else:
        utensil_dirs = [os.path.join(traj_lib_dir, workstation_name, filter_utensil)]
    utensil_container_dirs = []
    for u in utensil_dirs:
        u_path = os.path.join(traj_lib_dir, workstation_name, u)
        if not os.path.isdir(u_path):
            continue
        print(u)
        u_subdirs = os.listdir(u_path)
        for us in u_subdirs:
            us_path = os.path.join(u_path, us)
            if not os.path.isdir(us_path):
                continue
            if 'get_stuff' in os.listdir(us_path):
                print('   '+us)
                utensil_container_dirs.append(us_path)
                
    # print(utensil_container_dirs)
    return utensil_container_dirs

def get_urdf_path(workstation_name='greentown_candy'):
    home = os.path.expanduser('~')
    traj_lib_dir = os.path.join(home, 'traj_lib')
    urdf_dir = os.path.join(traj_lib_dir, workstation_name, 'urdf')
    urdf_name = '.' + workstation_name + '_utensils_in_bins-drake.urdf'
    urdf_path = os.path.join(urdf_dir, urdf_name)
    return urdf_path

def get_container_origin_xyz(urdf_path, container_dir):
    tree = RigidBodyTree(urdf_path)

    # set robot configuration
    q = np.zeros(7) # arm position doesn't matter here so just use zeros

    # look up indices of frames
    base_idx = tree.FindBody('base_link').get_body_index()
    container_idx = tree.FindBody(os.path.basename(container_dir)).get_body_index()

    # do kinematics with oint configuration
    kinsol = tree.doKinematics(q)
    # find relative transform between frames
    T = tree.relativeTransform(kinsol, base_idx, container_idx)
    return T[:3,3]

def get_workstation_config(workstation_name='greentown_candy'):
    '''
    Return loaded YAML of workstation config
    '''
    home = os.path.expanduser('~')
    traj_lib_dir = os.path.join(home, 'traj_lib')
    config_filename = workstation_name + '_config.yaml'
    config_path = os.path.join(traj_lib_dir, workstation_name, config_filename)
    with open(config_path, 'r') as stream:
        wp = yaml.load(stream)
    return wp

def get_container_dimensions_map(workstation_config):
    type_dim_map = {}
    container_dim_map = {}
    for t in workstation_config['container_types']:
        type_dim_map[t['container_type']] = t['container_dimensions']
    for i in workstation_config['container_instances']:
        container_dim_map[i['container_frame']] = type_dim_map[i['container_type']]
    return container_dim_map
        
def get_container_dimensions(workstation_config, container_dir):
    container_name = os.path.basename(container_dir)
    container_dim_map = get_container_dimensions_map(workstation_config)
    if container_name in container_dim_map:
        return container_dim_map[container_name]
    else:
        return None
    
def get_container_origin_and_dimensions(workstation_name, container_dir):
    urdf_path = get_urdf_path(workstation_name)
    wp = get_workstation_config(workstation_name)
    origin = get_container_origin_xyz(urdf_path, container_dir)
    dimensions = get_container_dimensions(wp, container_dir)
    return origin, dimensions

#######################################################################################################
################## THIS SECTION CONTAINS FUNCTIONS ANALYZING SCOOP ACTIONS ############################
#######################################################################################################

def get_container_action_index_map_filepath(container_dir):
    '''
    TODO @syler : read from workstation config
    '''
    container_string = os.path.basename(container_dir)
    index_dir = os.path.join(container_dir, 'get_stuff')
    index_map_filename = container_string + '_get_stuff_index_map.mpac'
    index_map_filepath = os.path.join(index_dir, index_map_filename)
    return index_map_filepath

def unpack_scoop_header(packed_header, container_dir, header_only=True):
    unpacked_header = {
        'id': packed_header[0],
        'scoop': unpack_scoop(packed_header[1]),
        'measured_result_map': packed_header[2],
        'modeled_result_map': packed_header[3]
    }
    if not header_only:
        traj_dir = os.path.join(container_dir, 'get_stuff', 'traj')
        scoop_filepath = os.path.join(traj_dir, unpacked_header['id']) + '.mpac'
        
        try:
            with open(scoop_filepath, 'rb') as ff:
                scoop_mpac = mp.load(ff)
            
            unpacked_header['scoop'] = unpack_scoop(scoop_mpac)
        except IOError:
            print('Scoop file', scoop_filepath, 'not found!')
    return unpacked_header

def load_scoops_from_action_index_map(container_dir, header_only=True):
    '''
    Load a msgpack action index map file and return a list of scoop dictionaries
    '''
    indexed_scoop_list = []
    action_index_map_filepath = get_container_action_index_map_filepath(container_dir)
    print(action_index_map_filepath)
    with open(action_index_map_filepath, 'rb') as ff:
        imap = mp.load(ff)
        
    for packed_indexed_scoop in imap.values():
        unpacked_indexed_scoop = unpack_scoop_header(packed_indexed_scoop, container_dir, header_only=header_only)
        indexed_scoop_list.append(unpacked_indexed_scoop)
        
    return indexed_scoop_list
        
def has_failed(indexed_scoop):
    for result in indexed_scoop['measured_result_map'].values():
        if result == 0.0:
            return True
    return False

def has_succeeded(indexed_scoop):
    for result in indexed_scoop['measured_result_map'].values():
        if result > 0.0:
            return True
    return False

#######################################################################################################
####### THIS SECTION CONTAINS FUNCTIONS DEALING WITH LISTS OF SCOOPS (modify description later) #######
#######################################################################################################

def get_list_from_scoops(indexed_scoop_list, field_name='entry', no_failed_scoops=False, only_failed_scoops=False, only_successful_scoops=False):
    # sanity check that we didn't do something wrong
    if only_failed_scoops:
        assert(only_failed_scoops != only_successful_scoops)
    if only_failed_scoops or no_failed_scoops:
        assert(only_failed_scoops != no_failed_scoops)

    entry_xyz_list = []
    for indexed_scoop in indexed_scoop_list:
        if no_failed_scoops and has_failed(indexed_scoop):
            continue
        if only_failed_scoops and not has_failed(indexed_scoop):
            continue
        if only_successful_scoops and not has_succeeded(indexed_scoop):
            continue
        entry_xyz_list.append(indexed_scoop['scoop'][field_name][1][0])
    return entry_xyz_list

def draw_scoop_points(x, y, z, ax, color='g', size=100, random_offset=False):
    '''
    Draw a scoop point on a 3D plot. 
    Optionally add random offset to easily visualize overlapping scoops.
    '''
    if random_offset:
        x = x + random.uniform(0, 0.005)
        y = y + random.uniform(0, 0.005)
        z = z + random.uniform(0, 0.005)
    ax.scatter(x, y, z, color=color, s=size)
    
def get_scoop_list_from_container(container_dir, header_only=False):
    indexed_scoop_list = load_scoops_from_action_index_map(container_dir, header_only=header_only)
    return indexed_scoop_list

def get_xyz_list_from_container(container_dir, field_name='entry', no_failed_scoops=False, only_failed_scoops=False, only_successful_scoops=False):
    indexed_scoop_list = get_scoop_list_from_container(container_dir)
    xyz_list = get_list_from_scoops(indexed_scoop_list, field_name=field_name, no_failed_scoops=no_failed_scoops, only_failed_scoops=only_failed_scoops, only_successful_scoops=only_successful_scoops)
    return xyz_list

#######################################################################################################
################# THIS SECTION CONTAINS THE HELPER FUNCTIONS FOR PLOTTING #############################
#######################################################################################################

def plot_scoop_xyz(xyz_list, ax, color='g', random_offset=False):
    '''
    Plots all points in xyz_list, a list of 3D points. 
    '''
    for point in xyz_list:
        draw_scoop_points(*point, ax, size=10, color=color, random_offset=random_offset)

def plot_scoop_xy(xy_list, ax, color='g', size=100, random_offset=False):
    '''
    Plots all points in xy_list, a list of 2D points. 
    '''
    for point in xy_list:
        ax.scatter(point[0], point[1], color=color, s=size)

def draw_container(origin, dimensions, ax):
    dx = dimensions[0]
    dy = dimensions[1]
    dz = dimensions[2]
    x0 = origin[0] - dx/2
    y0 = origin[1] - dy/2
    z0 = origin[2]
    x = [x0, x0+dx]
    y = [y0, y0+dy]
    z = [z0, z0+dz]
    for s, e in combinations(np.array(list(product(x, y, z))), 2):
        # print(np.sum(np.abs(s-e)))
        if (np.round(np.sum(np.abs(s-e)),2) == np.round(np.abs(dx),2)) or (np.round(np.sum(np.abs(s-e)),2) == np.round(np.abs(dy),2)) or (np.round(np.abs(np.sum(np.abs(s-e))),2) == np.round(np.abs(dz),2)):
            ax.plot3D(*zip(s, e), color='k')
            # print('found edge!')
            
def draw_container_2d(origin, dimensions, ax):
    dx = dimensions[0]
    dy = dimensions[1]
    x0 = origin[0] - dx/2
    y0 = origin[1] - dy/2
    x = [x0, x0+dx]
    y = [y0, y0+dy]
    for s, e in combinations(np.array(list(product(x, y))), 2):
        # print(np.sum(np.abs(s-e)))
        if (np.round(np.sum(np.abs(s-e)),2) == np.round(np.abs(dx),2)) or (np.round(np.sum(np.abs(s-e)),2) == np.round(np.abs(dy),2)):
            ax.plot(*zip(s, e), color='k')
            # print('found edge!')
            
def axis_equal_3d(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

#######################################################################################################
##################### THIS SECTION CONTAINS THE "MAIN" FUNCTIONS FOR PLOTTING #########################
#######################################################################################################

def plot_3D_separate():
    '''
         Plots (in 3D) entry (orange), key (yellow but looks green?), and exit points (cyan) for each container separately.
         Saves the plots in .png files.
    '''
    workstation_name = 'greentown_candy'
    container_dirs = get_utensil_container_dirs(workstation_name, filter_utensil='disher_2oz')

    random_offset = False
    for container_dir in container_dirs:
        container_name = os.path.basename(container_dir)

        origin, dims = get_container_origin_and_dimensions(workstation_name, container_dir)
        if dims is None:
            continue
        
        entry_xyz_list = get_xyz_list_from_container(container_dir, 'entry', no_failed_scoops=False, only_failed_scoops=False, only_successful_scoops=False)
        key0_xyz_list = get_xyz_list_from_container(container_dir, 'key0', no_failed_scoops=False, only_failed_scoops=False, only_successful_scoops=False)
        exit_xyz_list = get_xyz_list_from_container(container_dir, 'exit', no_failed_scoops=False, only_failed_scoops=False, only_successful_scoops=False)
        
        fig = plt.figure(num=None, figsize=(12, 10), dpi=120, facecolor='w', edgecolor='k')
        ax = fig.gca(projection='3d')

        plot_scoop_xyz(entry_xyz_list, ax, color='orange', random_offset=random_offset)
        plot_scoop_xyz(key0_xyz_list, ax, color='y', random_offset=random_offset)
        plot_scoop_xyz(exit_xyz_list, ax, color='c', random_offset=random_offset)
        draw_container(origin, dims, ax)
        plt.title(container_name)
        plt.xlabel('x')
        plt.ylabel('y')

        d = datetime.datetime.today()
        date_string = d.strftime('%Y%m%d')
        plt.savefig(container_name + '_' + date_string + '.png')

def plot_3D_combined():
    '''
        Plots (in 3D) the entry points for all containers in the workstation on the same graph.
    '''
    workstation_name = 'greentown_candy'
    container_dirs = get_utensil_container_dirs(workstation_name, filter_utensil='disher_2oz')

    fig = plt.figure(num=None, figsize=(16, 13), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')

    for container_dir in container_dirs:
        print(os.path.basename(container_dir))
        
        origin, dims = get_container_origin_and_dimensions(workstation_name, container_dir)
        if dims is None:
            continue
            
        entry_xyz_list = get_xyz_list_from_container(container_dir, 'entry', no_failed_scoops=False, only_failed_scoops=False, only_successful_scoops=False)
        
        plot_scoop_xyz(entry_xyz_list, ax=ax, color='orange')
        draw_container(origin, dims, ax)

    plt.xlabel('x')
    plt.ylabel('y')
    axis_equal_3d(ax)
    d = datetime.datetime.today()
    date_string = d.strftime('%Y%m%d')
    plt.savefig('all_' + date_string + '.png')

def plot_2D_combined():
    workstation_name = 'greentown_candy'
    container_dirs = get_utensil_container_dirs(workstation_name, filter_utensil='disher_2oz')

    fig = plt.figure(num=None, figsize=(12, 13), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)

    for container_dir in container_dirs:
        print(os.path.basename(container_dir))
        origin, dims = get_container_origin_and_dimensions(workstation_name, container_dir)
        if dims is None:
            continue
        entry_xyz_list = get_xyz_list_from_container(container_dir, 'entry', no_failed_scoops=False, only_failed_scoops=False, only_successful_scoops=False)
        
        origin, dims = get_container_origin_and_dimensions(workstation_name, container_dir)
        dims_minus_radius = dims
        tool_radius = 0.0275
        dims_minus_radius[0] = dims_minus_radius[0] - tool_radius
        dims_minus_radius[1] = dims_minus_radius[1] - tool_radius

        plot_scoop_xy(entry_xyz_list, ax=ax, size=10, color='orange')
        draw_container_2d(origin, dims, ax)
        draw_container_2d(origin, dims_minus_radius, ax)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    plt.grid(True)

    d = datetime.datetime.today()
    date_string = d.strftime('%Y%m%d')
    plt.savefig('2d_all_' + date_string + '.png')



if __name__ == '__main__':
    # prbly change later, don't think we want this command line option; only need 3D combined?
    dim_type = input("Enter desired dimensions (2/3): ")
    plot_type = input("Enter desired plot type (combined/separate): ")

    
    # TODO: maybe make the workstation name an input variable for plot functions? Then have that be a command line option
    # TODO: deal with ax

    if dim_type == 3 and plot_type == 'separate':
        plot_3D_separate()
    
    elif dim_type == 3 and plot_type == 'combined':
        plot_3D_combined()

    elif dim_type == 2 and plot_type == 'combined':
        plot_2D_combined()
    
    


