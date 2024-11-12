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
import random

SYN_DRAW_PATH = "../contour-detect/SynDraw/build"

if __name__ == "__main__":

    collect_data = False
    generate_silhouette_lines = False
    recompute_all_construction_lines = False
    declutter_construction_lines = False
    npr_rendering = True
    display = [1000, 1000]
    radius = 1.4
    lambda_0 = 10
    lambda_1 = 2.5
    lambda_1 = 5
    lambda_2 = 5
    lambda_3 = 40

    theta = 00
    phi = 00
    radius = 00

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="", type=str, help="Onshape document url")
    parser.add_argument("--data_folder", default="", type=str, help="data_folder path")
    parser.add_argument("--data_file", default="", type=str, help="data_file path")
    parser.add_argument("--collect_data", default="false", type=str, help="Download data from onshape")
    parser.add_argument("--generate_silhouette_lines", default="false", type=str, help="Generate silhouette lines from SynDraw")
    parser.add_argument("--recompute_all_construction_lines", default="false", type=str, help="Generate all construction lines from CAD-sequence")
    parser.add_argument("--declutter_construction_lines", default="false", type=str, help="Decluttering step")
    parser.add_argument("--npr_rendering", default="true", type=str, help="Render drawing in a concept sketch style")
    parser.add_argument("--display", default=[512, 512], nargs=2, type=int, help="Image size of rendered normal map and drawing")
    parser.add_argument("--theta", default=theta, type=float, help="Spherical coordinate theta of camera position around the object")
    parser.add_argument("--phi", default=phi, type=float, help="Spherical coordinate phi of camera position around the object")
    parser.add_argument("--radius", default=radius, type=float, help="Radius of camera position around the object")
    parser.add_argument("--only_feature_lines", default="false", type=str, help="Only generate intermediate feature lines")
    parser.add_argument("--only_final_npr_lines", default="false", type=str, help="Only generate feature lines from final shape and contour lines")
    parser.add_argument("--keep_all_lines", default="false", type=str, help="Keep all lines")
    parser.add_argument("--include_fillet_lines", default="true", type=str, help="")
    parser.add_argument("--per_view_folder_prefix", default="", type=str, help="")
    parser.add_argument("--verbose", default="true", type=str, help="")
    parser.add_argument("--designer", default="Professional6", type=str, help="For style")
    parser.add_argument("--stylesheet_file", default="", type=str, help="stylesheet file path")
    parser.add_argument("--clean_rendering", default="false", type=str, help="")
    parser.add_argument("--baseline_1", default="false", type=str, help="")
    parser.add_argument("--baseline_0", default="false", type=str, help="")
    parser.add_argument("--prep_full_sketch", default="false", type=str, help="")
    parser.add_argument("--use_full_sketch", default="false", type=str, help="")
    parser.add_argument("--only_sketch_rendering", default="false", type=str, help="")
    parser.add_argument("--cut_non_visible_points", default="false", type=str, help="")
    parser.add_argument("--lambda_0", default=1.0, type=float, help="")
    parser.add_argument("--lambda_1", default=0.5, type=float, help="")
    parser.add_argument("--lambda_2", default=0.1, type=float, help="")
    parser.add_argument("--cad_sequence_rendering", default="false", type=str, help="Render drawing in a concept sketch style")
    parser.add_argument("--jitter_cam_pos", default="false", type=str, help="Render drawing in a concept sketch style")
    parser.add_argument("--optimize_viewpoint", default="false", type=str, help="Render drawing in a concept sketch style")
    parser.add_argument("--find_best_lambdas", default="false", type=str, help="Render drawing in a concept sketch style")
    parser.add_argument("--match_stroke_length", default="true", type=str, help="Render drawing in a concept sketch style")
    args = parser.parse_args()
    stroke_dataset_designer_name = args.designer
    stylesheet_designer_name = args.designer
    style_sheet_file_name = args.stylesheet_file
    collect_data = args.collect_data == "True" or args.collect_data == "true"
    generate_silhouette_lines = args.generate_silhouette_lines == "True" or args.generate_silhouette_lines == "true"
    recompute_all_construction_lines = args.recompute_all_construction_lines == "True" or args.recompute_all_construction_lines == "true"
    declutter_construction_lines = args.declutter_construction_lines == "True" or args.declutter_construction_lines == "true"
    npr_rendering = args.npr_rendering == "True" or args.npr_rendering == "true"
    include_fillet_lines = args.include_fillet_lines == "True" or args.include_fillet_lines == "true"
    only_feature_lines = args.only_feature_lines == "True" or args.only_feature_lines == "true"
    only_final_npr_lines = args.only_final_npr_lines == "True" or args.only_final_npr_lines == "true"
    keep_all_lines = args.keep_all_lines == "True" or args.keep_all_lines == "true"
    verbose = args.verbose == "True" or args.verbose == "true"
    clean_rendering = args.clean_rendering == "True" or args.clean_rendering == "true"
    baseline_1 = args.baseline_1 == "True" or args.baseline_1 == "true"
    baseline_0 = args.baseline_0 == "True" or args.baseline_0 == "true"
    prep_full_sketch = args.prep_full_sketch == "True" or args.prep_full_sketch == "true"
    use_full_sketch = args.use_full_sketch == "True" or args.use_full_sketch == "true"
    only_sketch_rendering = args.only_sketch_rendering == "True" or args.only_sketch_rendering == "true"
    cad_sequence_rendering = args.cad_sequence_rendering == "True" or args.cad_sequence_rendering == "true"
    cut_non_visible_points = args.cut_non_visible_points == "True" or args.cut_non_visible_points == "true"
    optimize_viewpoint = args.optimize_viewpoint == "True" or args.optimize_viewpoint == "true"
    find_best_lambdas = args.find_best_lambdas == "True" or args.find_best_lambdas == "true"
    match_stroke_length = args.match_stroke_length == "True" or args.match_stroke_length == "true"
    theta = args.theta
    phi = args.phi
    radius = args.radius
    display = [args.display[0], args.display[1]]
    lambda_0 = args.lambda_0
    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2


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

    if npr_rendering:
        print("NPR_RENDERING")
        start_time = time()

        if not declutter_construction_lines:
            final_edges_file_name = os.path.join(data_folder, edge_data)
            # final_edges_file_name = os.path.join(per_view_data_folder, "final_edges.json")
            with open(final_edges_file_name, "r") as fp:
                sketch_data = json.load(fp)
                sketch_dimensions = (sketch_data["canvas_width"], sketch_data["canvas_height"])
                final_edges_dict = sketch_data["strokes"]
            
            # shutil.move(final_edges_file_name, per_view_data_folder)

        #style_sheet_file_name = os.path.join("data/stylesheets/"+stylesheet_designer_name+".json")
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



        # # DEBUG
        # # normal map rendering
        # patches = utils.load_last_faces(data_folder)
        # normal_pixels = get_normal_map_single_mesh(patches, display, cam_pos, obj_center, up_vec)
        # plt.gcf().subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
        #                           bottom=0.0,
        #                           top=1.0)
        # plt.gca().imshow(normal_pixels)
        # plt.gca().invert_yaxis()
        # plt.axis("off")
        # plt.gcf().set_size_inches(display[0]/100, display[1]/100)
        # normal_map_name = os.path.join(training_data_folder, "normal_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        # plt.savefig(normal_map_name, dpi=100)
        # plt.close(plt.gcf())

        # get opacities
        # new_opacities = optimize_opacities(final_edges_dict, stylesheet, cam_pos, obj_center, up_vec, mesh, VERBOSE=False)

        # gets opacity for each line. but theyre all silouette anyway
        new_opacities = optimize_opacities(final_edges_dict, stylesheet)
        
        #print("new_opacities")
        #for i, n in enumerate(new_opacities):
        #    print(i, n)


        # syn_sketch = geometry_match(final_edges_dict, stylesheet, cam_pos, obj_center, up_vec, display, clean_rendering=clean_rendering)

        # this basically just adds some oversketching to our silouette
        syn_sketch = geometry_match(final_edges_dict, stylesheet, sketch_dimensions)
        

        for s_id in range(len(syn_sketch.strokes)):
            for p_id in range(len(syn_sketch.strokes[s_id].points_list)):
                syn_sketch.strokes[s_id].points_list[p_id].add_data("pressure", new_opacities[s_id])

        #fig, ax = plt.subplots(nrows=1, ncols=1)
        #fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
        #                    bottom=0.0,
        #                    top=1.0)
        #syn_sketch.display_strokes_2(fig, ax,
        #                             linewidth_data=lambda p: p.get_data("pressure")+0.5,
        #                             color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
        #                                                      for p in s.points_list])
        #ax.set_xlim(0, display[0])
        #ax.set_ylim(display[1], 0)
        #ax.set_aspect("equal")
        #ax.axis("off")
        #fig.set_size_inches(display[0]/100, display[1]/100)
        #plt.savefig(os.path.join(training_data_folder, "test_strokes.png"), dpi=100)
        #plt.close(fig)

        #subdivide_long_curves(syn_sketch, VERBOSE=True)
        # DEBUG

        # adding random perturbations to the straight strokes
        syn_sketch = perturbate_sketch(syn_sketch)

        # if verbose:
        #     syn_sketch_file_name = os.path.join(per_view_data_folder, "synthetic_sketch.svg")
        #     skio.save(syn_sketch, syn_sketch_file_name)
        #     os.system("rsvg-convert -f pdf " + syn_sketch_file_name + " > " + syn_sketch_file_name.replace("svg", "pdf"))

        # width = 972
        # scale_factor = np.minimum(width, display[0])/width

        #syn_sketch = utils.rescale_rotate_center(syn_sketch)
        #npr_sketch_file_name = os.path.join(per_view_data_folder, "synthetic_sketch_v2.svg")
    


        # aligning synthetic sketch strokes (syn_sketch) with real-world strokes (stroke_dataset) 
        # based on several criteria, including stroke smoothness, length, and opacity.
        npr_sketch = match_strokes(syn_sketch, stroke_dataset, opacity_profiles,
                                   opacity_threshold=0.1,
                                   straight_line_nearest_neighbor_range=[0, 10],
                                   target_smoothness=0.3,
                                #    scale_factor=scale_factor,
                                   optimize_stroke_length=match_stroke_length)
        
        # time_step = 0.05
        # time_counter = 0.0
        # for s_id, s in enumerate(npr_sketch.strokes):
        #     s.add_avail_data("time")
        #     for p in s.points_list:
        #         p.add_data("time", time_counter)
        #         time_counter += time_step
        #     time_counter += time_step
        #npr_sketch = utils.rescale_rotate_center(npr_sketch)
        # npr_sketch_file_name = os.path.join(per_view_data_folder, "npr_sketch.json")
        # skio.save(npr_sketch, npr_sketch_file_name)
        # npr_sketch_file_name = os.path.join(per_view_data_folder, "npr_sketch.png")
        # skio.save(npr_sketch, npr_sketch_file_name)
        #exit()

        for stroke in npr_sketch.strokes:
            # stroke.svg_color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"
            stroke.width = 2

        out_npr_name = "npr_" + obj_data_name + ".svg"
        npr_sketch_file_name = os.path.join(per_view_data_folder, out_npr_name)
        skio.save(npr_sketch, npr_sketch_file_name)
        os.system("rsvg-convert -f pdf " + npr_sketch_file_name + " > " + npr_sketch_file_name.replace("svg", "pdf"))
        print("saved sketch in ", npr_sketch_file_name)

        print("stylization_time", time() - start_time)


        # pen_width = 1.5*scale_factor
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
        #                     bottom=0.0,
        #                     top=1.0)
        # ax.set_xlim(0, display[0])
        # ax.set_ylim(display[1], 0)
        # plt.gca().invert_yaxis()
        # ax.axis("off")
        # fig.set_size_inches(display[0]/100, display[1]/100)
        # # old rendering pipeline
        # #npr_sketch.display_strokes_2(fig, ax,
        # #                             linewidth_data=lambda p: p.get_data("pressure")+0.5,
        # #                             color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
        # #                                                      for p in s.points_list])
        # # new rendering pipeline
        # for s_i, s in enumerate(npr_sketch.strokes):
        #     for p_id in range(len(s.points_list)-1):
        #         pressure = s.points_list[p_id+1].get_data("pressure")
        #         #print(pen_width*pressure)
        #         ax.plot([s.points_list[p_id].coords[0], s.points_list[p_id+1].coords[0]],
        #                 [s.points_list[p_id].coords[1], s.points_list[p_id+1].coords[1]],
        #                 c=(0, 0, 0, min(1.0, pressure)),
        #                 lw=pen_width*pressure)
        # obj_id = data_folder.split("/")[-1]
        # # npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        # npr_sketch_name = os.path.join(training_data_folder, "npr.png")
        # # npr_sketch_name = os.path.join("only_visible_sketches", "npr_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        # #npr_sketch_name = os.path.join(training_data_folder, str(s_i)+".png")

        # # plt.savefig(npr_sketch_name, dpi=100)

        # print("saved sketch in ", npr_sketch_name)
        # #plt.close(fig)
        # #exit()
        # #for s in syn_sketch.strokes:
        # #    l = np.array(s.linestring.linestring)
        # #    plt.plot(np.array(l)[:, 0], np.array(l)[:, 1], c="black")
        # #ax.set_aspect("equal")
        # #npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_professional3.png")
        # #npr_sketch_name = os.path.join(training_data_folder, "npr_"+str(theta)+"_"+str(phi)+"_"+str(radius)+".png")
        # # npr_sketch_name = os.path.join(training_data_folder, "ours_lambda1_"+str(lambda_1)+"_lambda2_"+str(lambda_2)+"_style_"+stroke_dataset_designer_name+".png")
        # # plt.savefig(npr_sketch_name, dpi=100)
        # plt.close(fig)
        # # try:
        # #     #obj_id = int(data_folder.split("/")[-1])
        # #     obj_id = data_folder.split("/")[-1]
        # #     out_name = os.path.join(training_data_folder, "out_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_professional3_npr.png")
        # #     out_name = os.path.join(training_data_folder, "out_"+str(obj_id)+"_"+str(theta)+"_"+str(phi)+"_"+str(radius)+"_npr.png")
        # #     os.system("montage "+npr_sketch_name+" "+normal_map_name+" -tile 2x1 -geometry +0+0 "+out_name)
        # # except:
        # #     print("data_folder ", data_folder, "is not an integer")

