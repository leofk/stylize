from distutils.command.clean import clean
from pyrr import Matrix44
import utils
from optimize_lambda_parameters import optimize_lambda_parameters
import polyscope as ps
import matplotlib.pyplot as plt
from render_training_data import get_normal_map_single_mesh
from time import time
import pickle
import json
import numpy as np
import os, sys
from get_intermediate_shapes import get_intermediate_shapes
from opacity_optimization import optimize_opacities
from onshape.call import delete_document
from draw_extrude import draw_extrude
from prepare_decluttering import filter_identical, filter_identical_bvh, prepare_decluttering_v2, extract_strokes_dict
from declutter_gurobi import declutter
from copy import deepcopy
from render_shapes import features_lines_to_svg, typed_feature_lines_to_svg, \
    typed_feature_lines_to_svg_successive, indexed_lines_to_svg
from line_rendering import geometry_match, match_strokes, get_stroke_dataset, subdivide_long_curves, perturbate_sketch, get_opacity_profiles
from pylowstroke.sketch_io import SketchSerializer as skio
from get_best_viewpoint import get_best_viewpoint
sys.setrecursionlimit(10000)
import shutil
import logging
logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="", type=str, help="data_folder path")
    parser.add_argument("--data_file", default="", type=str, help="data_file path")
    parser.add_argument("--designer", default="Professional6", type=str, help="For style")
    parser.add_argument("--stylesheet_file", default="", type=str, help="stylesheet file path")
    args = parser.parse_args()


    style_sheet_file_name = args.stylesheet_file
    stroke_dataset_designer_name = args.designer
    data_folder = args.data_folder
    edge_data = args.data_file

    obj_data_name = os.path.splitext(os.path.basename(edge_data))[0]

    # per_view_data_folder = os.path.join(data_folder, edge_data)
    # if not os.path.exists(per_view_data_folder):
    #     os.mkdir(per_view_data_folder)
    per_view_data_folder = data_folder

    # training_data_folder = os.path.join(per_view_data_folder, "training_data")
    # if not os.path.exists(training_data_folder):
    #     os.mkdir(training_data_folder)
    training_data_folder = per_view_data_folder

    print("NPR_RENDERING")
    start_time = time()

    final_edges_file_name = os.path.join(data_folder, edge_data)
    
    with open(final_edges_file_name, "r") as fp:
        sketch_data = json.load(fp)
        sketch_dimensions = (sketch_data["canvas_width"], sketch_data["canvas_height"])
        final_edges_dict = sketch_data["strokes"]
    
    with open(style_sheet_file_name, "r") as fp:
        stylesheet = json.load(fp)

    stroke_dataset_name = os.path.join("data/stroke_datasets", stroke_dataset_designer_name+".pkl")
    if os.path.exists(stroke_dataset_name):
        with open(stroke_dataset_name, "rb") as fp:
            stroke_dataset = pickle.load(fp)
    else:
        stroke_dataset = get_stroke_dataset(stroke_dataset_designer_name)
        with open(stroke_dataset_name, "wb") as fp:
            pickle.dump(stroke_dataset, fp)

    opacity_profiles_name = os.path.join("data/opacity_profiles", stroke_dataset_designer_name+".json")
    if os.path.exists(opacity_profiles_name):
        with open(opacity_profiles_name, "r") as fp:
            opacity_profiles = json.load(fp)
    else:
        opacity_profiles = get_opacity_profiles(stroke_dataset_designer_name)
        with open(opacity_profiles_name, "w") as fp:
            json.dump(opacity_profiles, fp)


    # gets opacity for each line. but theyre all silouette anyway
    new_opacities = optimize_opacities(final_edges_dict, stylesheet)
    
    #print("new_opacities")
    #for i, n in enumerate(new_opacities):
    #    print(i, n)


    # add some overksetching to contours
    syn_sketch = geometry_match(final_edges_dict, stylesheet, sketch_dimensions)
    

    for s_id in range(len(syn_sketch.strokes)):
        for p_id in range(len(syn_sketch.strokes[s_id].points_list)):
            syn_sketch.strokes[s_id].points_list[p_id].add_data("pressure", new_opacities[s_id])


    #subdivide_long_curves(syn_sketch, VERBOSE=True)
    # DEBUG

    # adding random perturbations to the straight strokes
    syn_sketch = perturbate_sketch(syn_sketch)

    # aligning synthetic sketch strokes (syn_sketch) with real-world strokes (stroke_dataset) 
    # based on several criteria, including stroke smoothness, length, and opacity.
    npr_sketch = match_strokes(syn_sketch, stroke_dataset, opacity_profiles,
                                opacity_threshold=0.1,
                                straight_line_nearest_neighbor_range=[0, 10],
                                target_smoothness=0.3,
                            #    scale_factor=scale_factor,
                                optimize_stroke_length=match_stroke_length)
    

    for stroke in npr_sketch.strokes:
        # stroke.svg_color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"
        stroke.width = 2

    out_npr_name = "npr_" + obj_data_name + ".svg"
    npr_sketch_file_name = os.path.join(per_view_data_folder, out_npr_name)
    skio.save(npr_sketch, npr_sketch_file_name)
    os.system("rsvg-convert -f pdf " + npr_sketch_file_name + " > " + npr_sketch_file_name.replace("svg", "pdf"))
    print("saved sketch in ", npr_sketch_file_name)

    print("stylization_time", time() - start_time)

