from time import time
import pickle
import json
import os, sys
from opacity_optimization import optimize_opacities
from line_rendering import geometry_match, geometry_match_dual, match_strokes, get_stroke_dataset, perturbate_sketch, get_opacity_profiles
from pylowstroke.sketch_io import SketchSerializer as skio
sys.setrecursionlimit(10000)
import logging
logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
import argparse

import json
from collections import defaultdict
import os

def modify_json(input_file_path):
    with open(input_file_path, 'r') as file:
        data = json.load(file)

        points = data.get("points", [])
        canvas_width = data["canvas"]["width"]
        canvas_height = data["canvas"]["height"]

        new_json_structure = {}
        stroke_dict = defaultdict(list)

        for point in points:
            stroke_dict[point["stroke_id"]].append(point)

        for stroke_id, points_list in stroke_dict.items():
            geometry = []
            for point in points_list:
                x = (point["x"] + 1.0) / 2.0 * canvas_width
                y = canvas_height - ((-point["y"] + 1.0) / 2.0 * canvas_height)
                geometry.append([x, y])

            typeline = "sketch"
            stroke_type = typeline  # Modify if needed
            line_type = typeline    # Modify if needed
            feature_id = 1
            labels = [{"type": stroke_type, "feature_id": feature_id}]
            visibility_score = 1.0
            stroke_id_str = str(stroke_id)  

            new_json_structure[stroke_id_str] = {
                "geometry": geometry,
                "type": stroke_type,
                "feature_id": feature_id,
                "original_labels": labels,
                "visibility_score": visibility_score,
                "id": stroke_id,
                "line_type": line_type
            }

    final_output_structure = {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "strokes": new_json_structure
    }

    return final_output_structure


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="", type=str, help="data_folder path")
    parser.add_argument("--data_file", default="", type=str, help="data_file path")
    
    parser.add_argument("--designer", default="Professional6", type=str, help="For style")
    parser.add_argument("--stylesheet_name", default="", type=str, help="stylesheet file path")
    parser.add_argument("--match_stroke_length", default="True", type=str, help="Render drawing in a concept sketch style")
    
    
    parser.add_argument("--dir_1",  default="", type=str, help="data_folder path")
    parser.add_argument("--file_1",  default="", type=str, help="data_folder path")

    parser.add_argument("--dir_2",  default="", type=str, help="data_folder path")
    parser.add_argument("--file_2",  default="", type=str, help="data_folder path")

    args = parser.parse_args()

    match_stroke_length = args.match_stroke_length == "True"
    stylesheet_name = args.stylesheet_name
    stroke_dataset_designer_name = args.designer

    input_prefix = "../scratch/synthetic/"
    output_prefix = "../scratch/npr/"

    intput_dir_1 = input_prefix + args.dir_1
    intput_dir_2 = input_prefix + args.dir_2
    
    output_dir_1 = output_prefix + args.dir_1
    output_dir_2 = output_prefix + args.dir_2

    file_name_1 = args.file_1
    file_name_2 = args.file_2


    print("NPR_RENDERING")
    start_time = time()

    final_edges_file_name = os.path.join(intput_dir_1, file_name_1 + ".json")
    sketch_data = modify_json(final_edges_file_name)
    sketch_dimensions = (sketch_data["canvas_width"], sketch_data["canvas_height"])
    final_edges_dict = sketch_data["strokes"]
    
    final_edges_file_name_2 = os.path.join(intput_dir_2, file_name_2 + ".json")
    sketch_data_2 = modify_json(final_edges_file_name_2)
    sketch_dimensions_2 = (sketch_data_2["canvas_width"], sketch_data_2["canvas_height"])
    final_edges_dict_2 = sketch_data_2["strokes"]
    

    style_sheet_file_name = os.path.join("data/stylesheets", stylesheet_name+".json")
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
    
    # syn_sketch = geometry_match(final_edges_dict, stylesheet, sketch_dimensions)
    syn_sketch, syn_sketch_2 = geometry_match_dual(final_edges_dict, final_edges_dict_2, stylesheet, sketch_dimensions)
    

    for s_id in range(len(syn_sketch.strokes)):
        for p_id in range(len(syn_sketch.strokes[s_id].points_list)):
            syn_sketch.strokes[s_id].points_list[p_id].add_data("pressure", new_opacities[s_id])

    for s_id in range(len(syn_sketch_2.strokes)):
        for p_id in range(len(syn_sketch_2.strokes[s_id].points_list)):
            syn_sketch_2.strokes[s_id].points_list[p_id].add_data("pressure", new_opacities[s_id])

    #subdivide_long_curves(syn_sketch, VERBOSE=True)
    # DEBUG

    # adding random perturbations to the straight strokes
    # syn_sketch = perturbate_sketch(syn_sketch)

    # aligning synthetic sketch strokes (syn_sketch) with real-world strokes (stroke_dataset) 
    # based on several criteria, including stroke smoothness, length, and opacity.
    npr_sketch, npr_sketch_2 = match_strokes(syn_sketch, syn_sketch_2,
                               stroke_dataset, opacity_profiles,
                                opacity_threshold=0.1,
                                straight_line_nearest_neighbor_range=[0, 10],
                                target_smoothness=0.3,
                            #    scale_factor=scale_factor,
                                optimize_stroke_length=match_stroke_length)
    

    # for stroke in npr_sketch.strokes:
    #     # stroke.svg_color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"
    #     stroke.width = 2

    # for stroke in npr_sketch_2.strokes:
    #     # stroke.svg_color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"
    #     stroke.width = 2

    os.makedirs(output_dir_1, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)

    out_npr_name = "npr_" + file_name_1 + ".svg"
    out_npr_name_2 = "npr_" + file_name_2 + ".svg"

    npr_sketch_file_name = os.path.join(output_dir_1, out_npr_name)
    npr_sketch_file_name_2 = os.path.join(output_dir_2, out_npr_name_2)


    skio.save(npr_sketch, npr_sketch_file_name)
    skio.save(npr_sketch_2, npr_sketch_file_name_2)

    # os.system("rsvg-convert -f pdf " + npr_sketch_file_name + " > " + npr_sketch_file_name.replace("svg", "pdf"))
    # os.system("rsvg-convert -f pdf " + npr_sketch_file_name_2 + " > " + npr_sketch_file_name_2.replace("svg", "pdf"))

    # print("saved sketch in ", npr_sketch_file_name)
    print("stylization_time", time() - start_time)

