# Install library
#-------------------------------------------------------------------------#
import os
import json
import nibabel as nib
import numpy as np
import EMPRISE

# Load data
#-------------------------------------------------------------------------#
"""
Modify the json file for your analysis
condition: digit, spoken, visual, audio1, audio2
code_dir: specify the directory containing "config.json" file.
"""
code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"

config_file = os.path.join(code_dir, "config.json")
with open(config_file) as f:
    config = json.load(f)
globals().update(config)

# Parameters
#-------------------------------------------------------------------------#
proj_dir = PROJECT_DIR
subject  = SUBJECT
session  = SESSION
model    = MODEL # model name
space    = SPACE
target   = "split" if CV else "all" # "all" or "split"

# start Session class
#-------------------------------------------------------------------------#
sess = EMPRISE.Session(subject, session)

# start Model class
#-------------------------------------------------------------------------#
mod = EMPRISE.Model(subject, session, model, space_id=space)

print(f'\n\n-> Subject "{mod.sub}", Session "{mod.ses}", Space "{mod.space}":')

mesh_files = mod.get_mesh_files(space, surface='pial')

# Load fsaverage surface file (GIFTI)
surf = nib.load(mesh_files['left'])  # or right.gii
coords = surf.darrays[0].data
faces = surf.darrays[1].data

# Compute surface area
def triangle_area(pts):
    a = pts[1] - pts[0]
    b = pts[2] - pts[0]
    return 0.5 * np.linalg.norm(np.cross(a, b))

areas = [triangle_area(coords[face]) for face in faces]
total_area = sum(areas)
n_vertices = coords.shape[0]
area_per_vertex = total_area / n_vertices

print(f"Average area per vertex: {area_per_vertex:.2f} mm²")


# threshold and cluster vertices from surface-based results
#-------------------------------------------------------------------------#
verts, trias = mod.threshold_AFNI_cluster(crit='Rsqmb,p=0.05B',mesh='pial',cv=False)

""" # Step 1: R-squared map thresholding
#---------------------------------------------------------------------#
hemis    = {'L': 'left', 'R': 'right'}
res_file = mod.get_results_file('L')
filepath = res_file[:res_file.find('numprf.mat')]
Rsq_str  = ['Rsq','cvRsq'][int(cv)]
Rsq_thr  = filepath + Rsq_str + '_thr-' + crit + '.surf.gii'

# display message
print('\n-> Subject "{}", Session "{}", Model "{}",\n   Space "{}", Surface "{}":'. \
        format(mod.sub, mod.ses, mod.model, mod.space, mesh))

# threshold maps
print('   - Step 1: threshold R-squared maps ... ', end='')
if not os.path.isfile(Rsq_thr):
    maps = mod.threshold_maps(crit, cv)
    # dictionary "maps":
    # - keys "mu", "fwhm", "beta", "Rsq"
    #   - sub-keys "left", "right"
    print()
else:
    print('already done.')
    maps  = {}
    paras = ['mu','fwhm','beta',Rsq_str]
    for para in paras:
        maps[para] = {}
        for hemi in hemis.keys():
            res_file = mod.get_results_file(hemi)
            filepath = res_file[:res_file.find('numprf.mat')]
            maps[para][hemis[hemi]] = filepath + para + '_thr-' + crit + '.surf.gii'


# Step 2: AFNI surface clustering
#---------------------------------------------------------------------#
cls_sh    = proj_dir + 'code/Shell/cluster_surface'
if mod.space == 'fsnative':  cls_sh = cls_sh + '.sh'
if mod.space == 'fsaverage': cls_sh = cls_sh + '_fsa.sh'
img_str   = 'space-' + mod.space + '_' + Rsq_str + '_thr-' + crit
mesh_file = mod.get_mesh_files(mod.space, surface=mesh)['left']
if mod.space == 'fsnative':
    anat_pref = mesh_file[mesh_file.find('sub-')+len('sub-000/'):mesh_file.find('hemi-L')]
if mod.space == 'fsaverage':
    anat_pref = mesh_file[mesh_file.find('fsaverage/')+len('fsaverage/'):mesh_file.find('left.gii')]
Rsq_cls   = filepath + Rsq_str + '_thr-' + crit + '_cls-' + 'SurfClust' + '.surf.gii'

# cluster surface
print('   - Step 2: surface cluster using AFNI ... ', end='')
if not os.path.isfile(Rsq_cls):
    print('\n')
    AFNI_cmd = 'AFNI {} {} {} {} {} {}'. \
                format(cls_sh, self.sub, self.ses, self.model, img_str, anat_pref)
    os.system(AFNI_cmd)
    print()
else:
    print('already done.')
 """