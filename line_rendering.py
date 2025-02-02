from pylowstroke.sketch_io import SketchSerializer as skio
import imageio
from skimage.measure import EllipseModel
from trimesh.registration import procrustes
import cv2
from pylowstroke.sketch_core import StrokePoint, Stroke
import seaborn as sns
from fitCurves import fitCurve, chordLengthParameterize, generate_bezier_without_tangents
from bezier import q as eval_bezier
from shapely.geometry import MultiPoint, LineString, Point
import json
from copy import deepcopy
from pylowstroke.sketch_core import Stroke, StrokePoint, Sketch
import numpy as np
import os, sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(10000)

# go from our labels to overshooting categories
overshoot_mapping = {
    "extrude_line": "scaffolds",
    "fillet_line": "scaffolds",
    "feature_line": "ridges visible",
    "grid_lines": "scaffolds",
    "sketch": "scaffolds",
    "outline": "silhouette smooth",
    "section_lines": "surfacing: cross section scaffold",
    "silhouette_line": "silhouette smooth",
    "circle_square_line": "scaffolds",
}

opacity_mapping = {
    "extrude_line": "scaffold",
    "fillet_line": "scaffold",
    "feature_line": "edges",
    "grid_lines": "scaffold",
    "sketch": "scaffold",
    "outline": "silhouette",
    "section_lines": "cross_sections",
    "silhouette_line": "silhouette",
    "circle_square_line": "scaffold",
}

label_names = {
    0: 'silhouette smooth' ,
    1: 'ridges visible' ,
    2: 'vallleys visible',
    3: 'ridges occluded',
    4: 'valleys occluded',
    5: 'discriptive cross sections',
    6: 'axis and grids',
    7: 'scaffolds',
    8: 'scaffolds: lines to vp',
    9: 'scaffolds: square for an ellipse',
    10: 'scaffolds: tangents',
    11: 'surfacing: cross section',
    12: 'surfacing: cross section scaffold',
    13: 'surfacing: mirroring',
    14: 'surfacing: temporary plane',
    15: 'surfacing: projection lines',
    16: 'proportions div. rectangle2',
    17: 'proportions div. rectangle3',
    18: 'proportions div. ellipse',
    19: 'proportions mult. rectangle',
    20: 'hinging and rotating elements',
    21: 'ticks',
    22: 'hatching',
    23: 'text',
    24: 'outline',
    25: 'shadow construction',
    26: 'background'
}
inv_label_names = {v: k for k, v in label_names.items()}

def apply_homography(pts, h):
    #print(pts)
    #print(h)
    new_pts = np.ones([len(pts), 3])
    new_pts[:, :2] = pts
    new_pts = np.dot(h, new_pts.T).T
    new_pts[:, :2] /= new_pts[:, -1].reshape(-1, 1)
    return new_pts

def match_two_strokes_homography(s, other_s, VERBOSE=False):
    #s_points = np.array([p.coords for p in s.points_list])
    #s_mean = np.mean(s_points, axis=0)
    #print(s_points)
    other_s_points = np.array([other_s.eval_point(t).coords for t in np.linspace(0, 1, num=10)])
    s_points = np.array([s.eval_point(t).coords for t in np.linspace(0, 1, num=len(other_s_points))])
    h, status = cv2.findHomography(other_s_points, s_points)
    #print(status)
    if np.any(status == 0):
        return None

    #other_s_points = np.array(other_s.linestring.linestring)
    other_s_points = np.array(other_s.linestring.linestring.coords)
    other_s_copied = apply_homography(other_s_points, h)
    new_points = []
    for p_id, p in enumerate(other_s.points_list):
        new_p = StrokePoint(other_s_copied[p_id][0], other_s_copied[p_id][1])
        new_p.add_data("pressure", p.get_data("pressure"))
        new_points.append(new_p)
    other_s_copied_stroke = Stroke(new_points)
    return other_s_copied_stroke



def match_two_strokes(s, other_s, VERBOSE=False):
    s_points = np.array([p.coords for p in s.points_list])
    s_mean = np.mean(s_points, axis=0)
    other_s_points = np.array([p.coords for p in other_s.points_list])
    other_s_points = np.array([other_s.eval_point(t).coords for t in np.linspace(0, 1, num=10)])
    s_points = np.array([s.eval_point(t).coords for t in np.linspace(0, 1, num=len(other_s_points))])
    trans_stroke = Stroke([])
    trans_stroke.from_array(other_s_points)
    trans_stroke.resample_rdp(epsilon=s.length()/10.0)
    scaling_factor = s.length()/trans_stroke.length()
    other_s_transformed_mean = np.mean(other_s_points, axis=0)
    other_s_points -= other_s_transformed_mean
    other_s_points *= scaling_factor
    length_ratio = LineString(other_s_points).length/s.length()
    if length_ratio < 0.9 or length_ratio > 1.1:
        return None
    other_s_points += s_mean# - other_s_mean
    tmp_pts = np.ones([len(other_s_points), 3])
    tmp_pts[:, :2] = other_s_points
    tmp_s_pts = np.ones([len(s_points), 3])
    tmp_s_pts[:, :2] = s_points
    transformation, other_s_transformed, _ = procrustes(tmp_pts, tmp_s_pts,
                                                        scale=False)
    #other_s_transformed, transformation, angle = utils.icp_registration(
    #    other_s_points, s_points, with_scaling=False)
    other_s_stroke = Stroke([])
    other_s_stroke.from_array(other_s_points)
    other_s_stroke.resample_rdp(epsilon=s.length()/10.0)
    tmp_pts = np.ones([len(other_s.points_list), 4])
    tmp_pts[:, :2] = scaling_factor*(np.array([p.coords for p in other_s.points_list])-other_s_transformed_mean) + s_mean
    tmp_pts = np.dot(transformation, tmp_pts.T).T
    tmp_pts[:, :3] /= tmp_pts[:, 3].reshape(-1, 1)
    other_s_copied = tmp_pts[:, :2]
    #other_s_copied = utils.apply_icp_transformation(
    #    scaling_factor*(np.array([p.coords for p in other_s.points_list])-other_s_transformed_mean) + s_mean,
    #    transformation)
    new_points = []
    for p_id, p in enumerate(other_s.points_list):
        new_p = StrokePoint(other_s_copied[p_id][0], other_s_copied[p_id][1])
        new_p.add_data("pressure", p.get_data("pressure"))
        new_points.append(new_p)
    other_s_copied_stroke = Stroke(new_points)
    if VERBOSE:
        plt.plot(s_points[:, 0], s_points[:, 1], c="blue")
        plt.plot(other_s_points[:, 0], other_s_points[:, 1], c="red")
        plt.plot(other_s_copied[:, 0], other_s_copied[:, 1], c="cyan")
        plt.axis("equal")
        plt.show()
    return other_s_copied_stroke

def match_two_ellipses(s1, s2, VERBOSE=False):
    s1_points = np.array([p.coords for p in s1.points_list])
    s1_mean = np.mean(s1_points, axis=0)
    s2_points = np.array([p.coords for p in s2.points_list])
    s2_mean = np.mean(s2_points, axis=0)

    s2_new_points = deepcopy(s2_points) - s2_mean
    _, s1_u, s1_s = s1.get_ellipse_fitting()
    _, s2_u, s2_s = s2.get_ellipse_fitting()
    s1_angle = np.arccos(s1_u[0, 1])
    s2_angle = np.arccos(s2_u[0, 1])
    if VERBOSE:
        print(s1_u)
        print(s1_s)
        print(s2_u)
        print(s2_s)
        print(s1_angle)
        print(s2_angle)
    s1_u_inv = np.array([[-np.sin(-s1_angle), np.cos(-s1_angle)],
                         [np.cos(-s1_angle), -np.sin(-s1_angle)]])
    s2_u_inv = np.array([[-np.sin(-s2_angle), np.cos(-s2_angle)],
                         [np.cos(-s2_angle), -np.sin(-s2_angle)]])
    transform = np.sqrt(float(len(s2_new_points))/2.0) * np.matmul(np.diag(1/s2_s), s2_u_inv)  # transformation matrix
    s2_new_points = transform.dot(s2_new_points.T).T
    if VERBOSE:
        plt.plot(s2_new_points[:, 0], s2_new_points[:, 1], c="red")
    transform = np.sqrt(2.0 / float(len(s1_points))) * np.matmul(s1_u_inv, np.diag(s1_s))  # transformation matrix
    transform = np.sqrt(2.0 / float(len(s1_points))) * np.matmul(s1_u, np.diag(s1_s))  # transformation matrix
    s2_new_points = transform.dot(s2_new_points.T).T + s1_mean
    if VERBOSE:
        plt.plot(s1_points[:, 0], s1_points[:, 1], c="blue")
        plt.plot(s2_points[:, 0], s2_points[:, 1], c="red")
        plt.plot(s2_new_points[:, 0], s2_new_points[:, 1], c="cyan")
        plt.axis("equal")
        plt.show()
    new_points = []
    for p_id, p in enumerate(s2.points_list):
        new_p = StrokePoint(s2_new_points[p_id][0], s2_new_points[p_id][1])
        new_p.add_data("pressure", p.get_data("pressure"))
        new_points.append(new_p)
    s2_new = Stroke(new_points)
    return s2_new

def overshoot_stroke(s, ratio=0.1):
    #return s
    if s.is_curved():
        return s
    vec = s.points_list[1].coords - s.points_list[0].coords
    vec /= np.linalg.norm(vec)
    new_s = Stroke([])
    p_0 = deepcopy(s.points_list[0].coords)
    p_1 = deepcopy(s.points_list[-1].coords)
    p_0 -= np.random.uniform(0.0, ratio*s.length())*vec
    p_1 += np.random.uniform(0.0, ratio*s.length())*vec
    new_s.from_array([p_0, p_1])
    new_s.points_list[0].add_data("pressure", 1.0)
    new_s.points_list[1].add_data("pressure", 1.0)
    return new_s


def match_strokes(syn_sketch, syn_sketch_2,
                  stroke_dataset, opacity_profiles,
                  opacity_threshold=0.20,
                  straight_line_nearest_neighbor_range=[5, 15],
                  optimize_stroke_length=False,
                  target_smoothness=0.0,
                  scale_factor=1.0,
                  VERBOSE=False):
    

    new_sketch = Sketch()
    new_sketch.height = syn_sketch.height
    new_sketch.width = syn_sketch.width
    new_sketch.strokes = []

    new_sketch_2 = Sketch()
    new_sketch_2.height = syn_sketch.height
    new_sketch_2.width = syn_sketch.width
    new_sketch_2.strokes = []

    """ SETUP DATA """

    # stroke_dataset_opacities = np.array([s.get_mean_data("pressure") for s in stroke_dataset])
    # median_opacities = [o["median_opacity"] for o in opacity_profiles]
    median_opacities = []
    for o in opacity_profiles:
        if len(o["opacities"]) > 1:
            #median_opacities.append(np.mean(np.array(o["opacities"])[:, 1]))
            median_opacities.append(np.max(np.array(o["opacities"])[:, 1]))
        else:
            median_opacities.append(-1.0)

    smoothness_terms = np.array([s.smoothness for s in stroke_dataset])
    
    straight_line_ids = []
    curved_line_ids = []
    for s_id, s in enumerate(stroke_dataset):
        #if len(s.points_list) < 10:
        #    continue
        try:
            if not s.is_curved():
                straight_line_ids.append(s_id)
            elif not s.is_ellipse():
                curved_line_ids.append(s_id)
        except:
            continue
    straight_line_ids = np.array(straight_line_ids)
    curved_line_ids = np.array(curved_line_ids)

    # recompute smoothness for straight lines
    for s_id in straight_line_ids:
        s = stroke_dataset[s_id]
        smoothness_terms[s_id] = compute_smoothness_straight(s)

    if optimize_stroke_length:
        stroke_lengths = np.array([s.length() for s in stroke_dataset])
    smoothness_terms -= np.min(smoothness_terms)
    smoothness_terms /= np.max(smoothness_terms)

    smoothness_terms = np.abs(target_smoothness - smoothness_terms)
    smoothness_terms -= np.min(smoothness_terms)
    smoothness_terms /= np.max(smoothness_terms)

    ellipses_ids = [s_id for s_id, s in enumerate(stroke_dataset)
                    if len(s.points_list) > 3 and s.is_ellipse()]


    """ FIND MATCHING STYLE """
    
    for s_id, s in enumerate(syn_sketch.strokes):

        s2 = syn_sketch_2.strokes[s_id]

        if optimize_stroke_length:
            stroke_distance = np.abs((1.0/scale_factor)*s.length() - stroke_lengths)
            stroke_distance -= np.min(stroke_distance)
            stroke_distance /= np.max(stroke_distance)
            stroke_similarity_measure = 0.5*smoothness_terms + 0.5*stroke_distance
        opacity_multiplier = 1.0

        close_ids = []

        num_best=50
        topk=20

        if optimize_stroke_length:
            # close_ids = straight_line_ids[np.random.choice(np.argsort(stroke_similarity_measure[straight_line_ids])[:num_best], topk)] # LEO - REMOVED RANDOM SAMPLE
            close_ids = straight_line_ids[np.argsort(stroke_similarity_measure[straight_line_ids])[:num_best]]
            # print(f"stroke {s_id} is a LINE.\n")
        else:
            # close_ids = straight_line_ids[np.random.choice(np.argsort(smoothness_terms[straight_line_ids])[:num_best], topk)]# LEO - REMOVED RANDOM SAMPLE
            close_ids = straight_line_ids[np.argsort(smoothness_terms[straight_line_ids])[:num_best]]
        
        if s.is_curved():

            if optimize_stroke_length:
                # print(f"stroke {s_id} is a CURVE.\n")

                # close_ids = curved_line_ids[np.random.choice(np.argsort(stroke_similarity_measure[curved_line_ids])[:num_best], topk)]# LEO - REMOVED RANDOM SAMPLE
                close_ids = curved_line_ids[np.argsort(stroke_similarity_measure[curved_line_ids])[:num_best]]
            else:
                # print(f"stroke {s_id} is a ELLIPSE.\n")
                # close_ids = curved_line_ids[np.random.choice(np.argsort(smoothness_terms[curved_line_ids])[:num_best], topk)] # LEO - REMOVED RANDOM SAMPLE
                close_ids = curved_line_ids[np.argsort(smoothness_terms[curved_line_ids])[:num_best]] 
        
        if s.is_ellipse():
            close_ids = ellipses_ids


        """ TRANSFORM ORIGINAL TO MATCHED STYLE"""

        overshooted_stroke = deepcopy(s)
        overshooted_stroke_2 = deepcopy(s2)

        distances = []
        kept_ids = []

        for other_s_id in close_ids:
            other_s = stroke_dataset[other_s_id]

            if (not s.is_curved() and other_s.is_curved()) or (s.is_curved() and not other_s.is_curved()):
                continue
            if not s.is_ellipse() and other_s.is_ellipse():
                continue
            if len(other_s.points_list) < 2:
                continue
            if not s.is_curved():
                new_other_s = match_two_strokes(overshooted_stroke, other_s)
            elif not s.is_ellipse():
                new_other_s = match_two_strokes_homography(overshooted_stroke, other_s)
            else:
                new_other_s = match_two_ellipses(overshooted_stroke, other_s)
            if new_other_s is None:
                continue

            dist = new_other_s.linestring.linestring.hausdorff_distance(s.linestring.linestring)
            distances.append(dist)
            kept_ids.append(other_s_id)
            
        if len(kept_ids) == 0:
            continue

        min_distances_ids = np.argsort(distances)

        min_dist_thresh = min(len(distances), straight_line_nearest_neighbor_range[0])

        max_dist_thresh = straight_line_nearest_neighbor_range[1]

        if s.is_curved():
            min_dist_thresh = 0
            max_dist_thresh = 10
            if s.is_ellipse():
                max_dist_thresh = 5

        
        # new_s_id = kept_ids[np.random.choice(min_distances_ids[min_dist_thresh:max_dist_thresh], 1)[0]]
        new_s_id = kept_ids[min_distances_ids[min_dist_thresh:max_dist_thresh][0]]

        if not s.is_curved():
            new_other_s = match_two_strokes(overshooted_stroke, stroke_dataset[new_s_id])
            new_other_s_2 = match_two_strokes(overshooted_stroke_2, stroke_dataset[new_s_id])

        elif not s.is_ellipse():
            new_other_s = match_two_strokes_homography(overshooted_stroke, stroke_dataset[new_s_id])
            new_other_s_2 = match_two_strokes_homography(overshooted_stroke_2, stroke_dataset[new_s_id])

        else:
            new_other_s = match_two_ellipses(overshooted_stroke, stroke_dataset[new_s_id])
            new_other_s_2 = match_two_ellipses(overshooted_stroke_2, stroke_dataset[new_s_id])


            opacity_multiplier = stroke_dataset[new_s_id].opacity_multiplier

        mean_opacity = np.mean([p.get_data("pressure") for p in s.points_list])
        mean_opacity = opacity_multiplier*mean_opacity
        closest_opacity_profile_id = np.argmin(np.abs(median_opacities - mean_opacity))
        closest_opacity_profile = np.array(opacity_profiles[closest_opacity_profile_id]["opacities"])


        def process_geometry(stroke, closest_opacity_profile, mean_opacity):

            coords = np.array([p.coords for p in stroke.points_list])
            coords_ts = []
            line = LineString(coords)
            
            for p_id in range(len(coords)):
                p = coords[p_id]
                t = line.project(Point(p), normalized=True)
                coords_ts.append(t)
            coords_ts = np.array(coords_ts)
            
            complex_coords = [complex(p[0], p[1]) for p in coords]
            new_coords = np.array([
                [p.real, p.imag] for p in np.interp(
                    closest_opacity_profile[:, 0], coords_ts, complex_coords
                )
            ])
            all_ts = np.array(coords_ts.tolist() + closest_opacity_profile[:, 0].tolist())
            all_coords = np.array(coords.tolist() + new_coords.tolist())
            new_all_coords = all_coords[np.argsort(all_ts)]
            new_opacities = np.interp(
                np.sort(all_ts), closest_opacity_profile[:, 0], closest_opacity_profile[:, 1]
            )
            
            stroke.from_array(new_all_coords)
            for p_id in range(len(stroke.points_list)):
                stroke.points_list[p_id].data["pressure"] = new_opacities[p_id]
                stroke.points_list[p_id].data["pressure"] = mean_opacity
            
            return stroke


        new_other_s = process_geometry(new_other_s, closest_opacity_profile, mean_opacity)
        new_other_s_2 = process_geometry(new_other_s_2, closest_opacity_profile, mean_opacity)

        new_sketch.strokes.append(new_other_s)
        new_sketch_2.strokes.append(new_other_s_2)


    for s in new_sketch.strokes:
        s.add_avail_data("pressure")

    for s in new_sketch_2.strokes:
        s.add_avail_data("pressure")

    return new_sketch, new_sketch_2


# stylesheet.keys():
# category_name: {"opacity"}, {"overshooting"}

def generate_sketch(edges, style_labels, stylesheet, dimensions, clean_rendering):
        sketch = Sketch()
        sketch.width = dimensions[0]
        sketch.height = dimensions[1]

        geometries = [edge["geometry"] for edge_id, edge in edges.items()]

        for l_id, l in enumerate(geometries):
            line_type = style_labels[l_id]
            opacity = 1.0  # Default opacity

            points = []
            for p in l:
                tmp_p = StrokePoint(x=p[0], y=p[1])
                tmp_p.add_data("pressure", opacity)
                points.append(tmp_p)
            s = Stroke(points)

            if not clean_rendering:
                opensketch_line_type = overshoot_mapping[line_type]
                key = inv_label_names[opensketch_line_type]
                if str(key) in stylesheet["overshooting_per_type"].keys():
                    overshooting_ratio = stylesheet["overshooting_per_type"][str(key)]
                else:
                    overshooting_ratio = 0.1
                s = overshoot_stroke(s, overshooting_ratio)

            for p in s.points_list:
                p.add_data("pressure", opacity)

            sketch.strokes.append(s)

        for s in sketch.strokes:
            s.add_avail_data("pressure")

        return sketch

def geometry_match_dual(edges, edges2, stylesheet, dimensions, clean_rendering=False, VERBOSE=False):
    

    # Style label computation based on edges
    max_feature_id = max(
        max(l["feature_id"] for l in edge["original_labels"])
        for edge in edges.values()
    )

    style_labels = []
    for edge_id, edge in edges.items():
        tmp_max_feature_id = max(l["feature_id"] for l in edge["original_labels"])
        all_line_types = [l["type"] for l in edge["original_labels"]]
        if tmp_max_feature_id == max_feature_id and "silhouette_line" in all_line_types and edge["visibility_score"] > 0.9:
            style_labels.append("silhouette_line")
        elif tmp_max_feature_id == max_feature_id and ("feature_line" in all_line_types or edge["type"] == "feature_line"):
            style_labels.append("feature_line")
        else:
            style_labels.append("grid_lines")

    # Generate sketches
    sketch1 = generate_sketch(edges, style_labels, stylesheet, dimensions, clean_rendering)

    # Apply style_labels from edges to edges2
    sketch2 = generate_sketch(edges2, style_labels, stylesheet, dimensions, clean_rendering)

    return sketch1, sketch2


# def geometry_match(edges, stylesheet, cam_pos, obj_center, up_vec, display, clean_rendering=False, VERBOSE=False):
def geometry_match(edges, stylesheet, dimensions, clean_rendering=False, VERBOSE=False):
    #print(edges.keys())
    len_edges = len(edges.keys())
    sketch = Sketch()

    sketch.width = dimensions[0]
    sketch.height = dimensions[1]

    geometries = []
    for edge_id in range(len(edges.keys())):
        edge = edges[str(edge_id)]
        #print(edge)
        #print(edge["line_type"])
        # geometries.append(edge["geometry_3d"])
        geometries.append(edge["geometry"])

    # find out "style line types"
    max_feature_id = 0
    for s_id in range(len(edges)):
        max_feature_id = max(max_feature_id, np.max([l["feature_id"] for l in edges[str(s_id)]["original_labels"]]))
    style_labels = []
    for edge_id in edges.keys():
        edge = edges[edge_id]
        tmp_max_feature_id = np.max([l["feature_id"] for l in edge["original_labels"]])
        all_line_types = [l["type"] for l in edge["original_labels"]]
        if tmp_max_feature_id == max_feature_id and "silhouette_line" in all_line_types and edge["visibility_score"] > 0.9:
            style_labels.append("silhouette_line")
            continue
        if tmp_max_feature_id == max_feature_id and ("feature_line" in all_line_types or edge["type"] == "feature_line"):
            style_labels.append("feature_line")
            continue
        style_labels.append("grid_lines")

    #print(style_labels)
    #print("max_feature_id", max_feature_id)
    #projected_lines = project_points(geometries, cam_pos=cam_pos, obj_center=obj_center)
    #print(cam_pos, obj_center)
    # projected_lines = project_lines_opengl(geometries, display, cam_pos, obj_center, up_vec)
    projected_lines = geometries
    # bbox_points = [p for l in projected_lines for p in l]
    # bbox = MultiPoint(bbox_points).bounds
    #print("sketch_bbox", bbox)
    #plt.show()
    #exit()

    #stylesheet["opacities_per_type"]["scaffold"]["mu"] = 0.2
    #stylesheet["opacities_per_type"]["edges"]["mu"] = 0.7
    #stylesheet["opacities_per_type"]["silhouette"]["mu"] = 0.9
    # sketch.height = abs(bbox[2] - bbox[0])
    # sketch.width = abs(bbox[3] - bbox[1])
    #print(projected_lines)
    for l_id, l in enumerate(projected_lines):
        line_type = edges[str(l_id)]["line_type"]
        line_type = style_labels[l_id]
        opacity = 1.0
        #if clean_rendering:
        #    opacity = 1.0
        #else:
        #    opacity = min(1.0, max(0.0, np.random.normal(loc=stylesheet["opacities_per_type"][opacity_mapping[line_type]]["mu"],
        #                                                 scale=stylesheet["opacities_per_type"][opacity_mapping[line_type]]["sigma"]/2, size=1)[0]))
        #    #print(line_type, opacity)
        #    #opacity = stylesheet["opacities_per_type"][opacity_mapping[line_type]]["mu"]
        points = []
        for p in l:
            tmp_p = StrokePoint(x=p[0], y=p[1])
            tmp_p.add_data("pressure", opacity)
            points.append(tmp_p)
        s = Stroke(points)
        if not clean_rendering:
            opensketch_line_type = overshoot_mapping[line_type]
            key = inv_label_names[opensketch_line_type]
            if str(key) in stylesheet["overshooting_per_type"].keys():
                overshooting_ratio = stylesheet["overshooting_per_type"][str(key)]
            else:
                overshooting_ratio = 0.1
            s = overshoot_stroke(s, overshooting_ratio)
        for p in s.points_list:
            p.add_data("pressure", opacity)
        sketch.strokes.append(s)
    # if VERBOSE:
    #     plt.rcParams["figure.figsize"] = (20, 10)
    #     plt.rcParams["image.interpolation"] = "antialiased"

    #     fig, axes = plt.subplots(nrows=1, ncols=1)
    #     fig.subplots_adjust(wspace=0.0, hspace=1.0, left=0.0, right=1.0,
    #                         bottom=0.0,
    #                         top=1.0)

    #     sketch.display_strokes_2(fig=fig, ax=axes,
    #                              linewidth_data=lambda p: p.get_data("pressure")+0.5,
    #                              color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
    #                                                       for p in s.points_list])
    #                              #linewidth_process=lambda s: 2)
    #     axes.set_xlim(0, sketch.width)
    #     axes.set_ylim(sketch.height, 0)
    #     axes.axis("equal")
    #     axes.axis("off")
    #     plt.show()

    # sketch.width = 1000
    # sketch.height = 1000
    for s in sketch.strokes:
        s.add_avail_data("pressure")
    return sketch

def get_stroke_dataset(designer_name, VERBOSE=False):
    use_sketches = ["bumps", "flange", "wobble_surface", "mixer", "tubes", "waffle_iron"]
    #use_sketches = ["flange"]
    strokes = []
    designer_folder = os.path.join("../sketches_json_first_viewpoint", designer_name)
    for sketch_name in os.listdir(designer_folder):
        if os.path.isfile(sketch_name):
            continue
        print(sketch_name)
        if not sketch_name in use_sketches:
            continue
        sketch = skio.load(os.path.join(designer_folder, sketch_name, "view1_concept.json"))
        all_points = np.array([np.array(p.coords) for s in sketch.strokes for p in s.points_list])
        for s_id, s in enumerate(sketch.strokes):
            # compute smoothness for curves
            s.smoothness = 10000.0
            if not s.is_ellipse:
                continue
            if len(s.points_list) < 2:
                continue
            if not s.is_curved(threshold=0.05) or s.is_ellipse():
                if s.is_ellipse():
                    #443 0.1568627450980392 0.3343108504398827

                    # compute apparent opacity
                    pressures = [p.get_data("pressure") for p in s.points_list]
                    median_pressure = np.median(pressures)
                    if VERBOSE:
                        print(np.mean(pressures))
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0,
                                        bottom=0.0,
                                        top=1.0)
                    sketch.display_strokes_2(fig, ax,
                                             linewidth_data=lambda p: p.get_data("pressure")+0.5,
                                             #linewidth_data=lambda p: median_pressure,
                                             #color_process=lambda s: [(0, 0, 0, median_pressure)
                                             color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
                                                                      for p in s.points_list],
                                             display_strokes=[s_id])
                    ax.set_xlim(0, sketch.width)
                    ax.set_ylim(sketch.height, 0)
                    ax.set_aspect("equal")
                    ax.axis("off")

                    my_dpi = 100.0
                    fig.set_size_inches(((np.max(all_points[:, 0])-np.min(all_points[:, 0])) / my_dpi,
                                         (np.max(all_points[:, 1])-np.min(all_points[:, 1])) / my_dpi))
                    plt.savefig(os.path.join("mask.png"))
                    plt.close(fig)

                    s_id_mask = imageio.imread("mask.png")
                    mask = np.any(s_id_mask != [255, 255, 255, 255], axis=2)
                    apparent_pressures = s_id_mask[mask][:, 0]
                    ell_pts = np.argwhere(np.any(s_id_mask != [255, 255, 255, 255], axis=2))
                    ell_pts = np.array([[p[1], p[0]] for p in ell_pts])

                    ell = EllipseModel()
                    #ell.estimate(np.array(s.linestring.linestring))
                    ell.estimate(ell_pts)
                    ell_pts_mp = MultiPoint(ell_pts)
                    xc, yc, a, b, theta_orig = ell.params
                    fitted_ellipse = np.array(ell.predict_xy(np.linspace(0, 2*np.pi, 360)))

                    if VERBOSE:
                        plt.imshow(s_id_mask)
                        plt.scatter(fitted_ellipse[:, 0], fitted_ellipse[:, 1])
                        plt.scatter(ell_pts[:, 0], ell_pts[:, 1], c="r", s=10, alpha=0.3)
                    max_opacities = []

                    for p_id in range(len(fitted_ellipse)-1):
                        vec = fitted_ellipse[p_id+1] - fitted_ellipse[p_id]
                        vec /= np.linalg.norm(vec)
                        ortho_vec = np.array([-vec[1], vec[0]])
                        plot_ortho_vec = np.array([fitted_ellipse[p_id]-b/5*ortho_vec, fitted_ellipse[p_id]+b/5*ortho_vec])
                        intersected_ell_pts = np.array(LineString(plot_ortho_vec).buffer(1.0).intersection(ell_pts_mp), dtype=int)
                        if len(intersected_ell_pts.shape) < 2:
                            continue
                        if VERBOSE:
                            plt.plot(plot_ortho_vec[:, 0], plot_ortho_vec[:, 1], alpha=0.1, c="green")
                            #plt.scatter(intersected_ell_pts[:, 0], intersected_ell_pts[:, 1])
                        intersected_ell_pts = np.array([[p[1], p[0]] for p in intersected_ell_pts])
                        intersected_opacities = np.array([s_id_mask[x, y][0] for x, y in intersected_ell_pts])
                        max_opacity = np.min(intersected_opacities)
                        max_opacity_pt = np.argmin(intersected_opacities)
                        if VERBOSE:
                            plt.scatter([intersected_ell_pts[max_opacity_pt][1]], [intersected_ell_pts[max_opacity_pt][0]], c="black")
                        max_opacities.append(1-max_opacity/255)

                    if VERBOSE:
                        plt.show()
                        #plt.hist(apparent_pressures, bins=100)
                        plt.hist(max_opacities, bins=100)
                        plt.show()
                    apparent_opacity = np.median(max_opacities)
                    if VERBOSE:
                        print(s_id, apparent_opacity, median_pressure)
                        print(apparent_opacity/median_pressure)
                    s.opacity_multiplier = median_pressure/apparent_opacity
                strokes.append(s)
                continue
            # here we only have non-ellipse curves
            pts = np.array(s.linestring.linestring.coords)
            bspline = generate_bezier_without_tangents(pts)
            errors = []

            for i, (point, u) in enumerate(zip(pts, chordLengthParameterize(pts))):
                dist = np.sqrt(np.linalg.norm(eval_bezier(bspline, u)-point)**2)
                errors.append(dist)
            mean_error = np.mean(errors)
            bezier_pts = np.array([eval_bezier(bspline, t) for t in np.linspace(0.0, 1.0, 10)])
            bezier_length = LineString(bezier_pts).length
            smoothness_term = mean_error/bezier_length
            s.smoothness = smoothness_term
            #print("smoothness_term", smoothness_term)
            #plt.plot(pts[:, 0], pts[:, 1], c="g")
            #plt.plot(bezier_pts[:, 0], bezier_pts[:, 1], c="r")
            #plt.show()
            strokes.append(s)
    return strokes

def compute_smoothness_straight(s):
    pts = np.array([p.coords for p in s.points_list])
    seg = LineString(s.get_segment_fitting())
    errors = [seg.distance(Point(p)) for p in pts]
    mean_error = np.mean(errors)
    s = mean_error/seg.length
    return s

def get_opacity_profiles(designer_name):
    use_sketches = ["bumps", "flange", "wobble_surface", "mixer", "tubes", "waffle_iron"]
    opacity_profiles = []
    designer_folder = os.path.join("../sketches_json_first_viewpoint", designer_name)
    for sketch_name in os.listdir(designer_folder):
        if os.path.isfile(sketch_name):
            continue
        print(sketch_name)
        if not sketch_name in use_sketches:
            continue
        sketch = skio.load(os.path.join(designer_folder, sketch_name, "view1_concept.json"))
        for s in sketch.strokes:
            opacity_profile = {"median_opacity": 0.0,
                               "opacities": []}
            opacities = []
            for p in s.points_list:
                o = p.get_data("pressure")
                t = s.linestring.linestring.project(Point(p.coords), normalized=True)
                opacity_profile["opacities"].append([t, o])
                opacities.append(o)
            if len(opacities) > 0:
                opacity_profile["median_opacity"] = np.median(opacities)
            opacity_profiles.append(opacity_profile)
    return opacity_profiles

def subdivide_long_curves(sketch, VERBOSE=False):
    insert_ids = []
    new_meta_strokes = []
    print(len(sketch.strokes))
    for s_id, s in enumerate(sketch.strokes):
        if not s.is_curved() or s.is_ellipse():
            continue
        pts = np.array(s.linestring.linestring)
        #axes.plot(pts[:, 0], pts[:, 1], c="blue", lw=3)
        bspline = fitCurve(pts, 1.0)
        #max_error = computeMaxError(pts, bspline, chordLengthParameterize(pts))
        #print("max_error", max_error)
        # plot ctrl pts
        bspline_cmap = sns.color_palette("Set1", n_colors=len(bspline))
        new_strokes = []
        if VERBOSE:
            fig, axes = plt.subplots(nrows=1, ncols=1)

            fig.subplots_adjust(wspace=0.0, hspace=1.0, left=0.0, right=1.0,
                                bottom=0.0,
                                top=1.0)

            sketch.display_strokes_2(fig=fig, ax=axes,
                                     linewidth_data=lambda p: p.get_data("pressure")+0.5,
                                     color_process=lambda s: [(0, 0, 0, p.get_data("pressure"))
                                                              for p in s.points_list])
        for ctrl_pts_id, ctrl_pts in enumerate(bspline):
            bezier = np.array([eval_bezier(ctrl_pts, t) for t in np.linspace(0.0, 1.0, 20)])
            new_points = []
            for b in bezier:
                p = StrokePoint(b[0], b[1])
                p.add_data("pressure", s.points_list[0].get_data("pressure"))
                #p.add_data("pressure", 1.0)
                new_points.append(p)
            new_s = Stroke(new_points)
            new_s.add_avail_data("pressure")
            new_strokes.append(new_s)
            if VERBOSE:
                axes.plot(bezier[:, 0], bezier[:, 1], c=bspline_cmap[ctrl_pts_id], lw=3)
                axes.scatter(np.array(ctrl_pts)[:, 0], np.array(ctrl_pts)[:, 1], c=bspline_cmap[ctrl_pts_id])
                axes.plot(np.array(ctrl_pts)[:, 0], np.array(ctrl_pts)[:, 1], c=bspline_cmap[ctrl_pts_id])
        if VERBOSE:
            axes.set_xlim(0, sketch.width)
            axes.set_ylim(sketch.height, 0)
            axes.axis("equal")
            axes.axis("off")
            plt.show()
        insert_ids.append(s_id)
        new_meta_strokes.append(new_strokes)
    for new_strokes_id, rev_insert_id in enumerate(reversed(insert_ids)):
        if rev_insert_id == len(sketch.strokes) - 1:
            sketch.strokes[-1] = new_meta_strokes[-1][0]
        else:
            sketch.strokes = sketch.strokes[:rev_insert_id] + new_meta_strokes[len(new_meta_strokes)-1-new_strokes_id] + sketch.strokes[rev_insert_id+1:]

    #print(len(sketch.strokes))
    #exit()
    

def perturbate_sketch(sketch):
    for s_id, s in enumerate(sketch.strokes):
        if s.is_curved():
            continue
        s_pts = np.array(s.linestring.linestring.coords)
        new_s_pts = np.array([
            p + np.random.uniform(-0.02*s.length(), 0.02*s.length(), 2)
            for p in s_pts])
        #s_center = np.mean(s_pts, axis=0)
        #rnd_angle = np.deg2rad(np.random.uniform(low=-2.0, high=2.0))
        #s_vec = s_pts[-1] - s_pts[0]
        #s_vec /= np.linalg.norm(s_vec)
        #rot_mat = np.array([[np.cos(rnd_angle), -np.sin(rnd_angle)],
        #                    [np.sin(rnd_angle), np.cos(rnd_angle)]])
        #rot_s_vec = np.dot(rot_mat, s_vec)
        #new_s_pts = np.array([s_center - rot_s_vec*s.length()/2,
        #                      s_center + rot_s_vec*s.length()/2])
        p0 = StrokePoint(new_s_pts[0][0], new_s_pts[0][1])
        p0.add_data("pressure", s.points_list[0].get_data("pressure"))
        p1 = StrokePoint(new_s_pts[1][0], new_s_pts[1][1])
        p1.add_data("pressure", s.points_list[1].get_data("pressure"))
        sketch.strokes[s_id] = Stroke([p0, p1])
        #s.points_list[0].set_coords(new_s_pts[0][0], new_s_pts[0][1])
        #s.points_list[1].set_coords(new_s_pts[1][0], new_s_pts[1][1])
        sketch.strokes[s_id].add_avail_data("pressure")
    return sketch

if __name__ == "__main__":

    import pickle
    stroke_dataset_designer_name = "Professional6"
    stroke_dataset = get_stroke_dataset(stroke_dataset_designer_name)
    stroke_dataset_name = os.path.join("data/stroke_datasets", stroke_dataset_designer_name+".pkl")
    for s in stroke_dataset:
        if s.is_ellipse():
            continue
    with open(stroke_dataset_name, "wb") as fp:
        pickle.dump(stroke_dataset, fp)
    exit()
    opacity_profiles = get_opacity_profiles(stroke_dataset_designer_name)
    opacity_profiles_name = os.path.join("data/opacity_profiles", stroke_dataset_designer_name+".json")
    if not os.path.exists(os.path.join("data/opacity_profiles")):
        os.mkdir(os.path.join("data/opacity_profiles"))
    with open(opacity_profiles_name, "w") as fp:
        json.dump(opacity_profiles, fp)
    exit()
    data_folder = "data/36"
    #file_name = os.path.join(data_folder, "final_drawing.svg")
    #syn_sketch = skio.load(file_name)
    file_name = os.path.join(data_folder, "decluttered_lambda0_10.json")
    style_sheet_file_name = os.path.join("../wires_python/data/stylesheets/student5.json")
    with open(style_sheet_file_name, "r") as fp:
        stylesheet = json.load(fp)
    with open(file_name, "rb") as fp:
        edges = json.load(fp)
    stroke_dataset = get_stroke_dataset("student8")
    stroke_dataset_designer_name = "Professional6"
    opacity_profiles = get_opacity_profiles(stroke_dataset_designer_name)
    opacity_profiles_name = os.path.join("data/opacity_profiles", stroke_dataset_designer_name+".json")
    if not os.path.exists(os.path.join("data/opacity_profiles")):
        os.mkdir(os.path.join("data/opacity_profiles"))
    with open(opacity_profiles_name, "w") as fp:
        json.dump(opacity_profiles, fp)
    exit()
    data_folder = "data/36"
    #file_name = os.path.join(data_folder, "final_drawing.svg")
    #syn_sketch = skio.load(file_name)
    file_name = os.path.join(data_folder, "decluttered_lambda0_10.json")
    style_sheet_file_name = os.path.join("../wires_python/data/stylesheets/student5.json")
    with open(style_sheet_file_name, "r") as fp:
        stylesheet = json.load(fp)
    with open(file_name, "rb") as fp:
        edges = json.load(fp)
    stroke_dataset = get_stroke_dataset("student8")
    syn_sketch = geometry_match(edges, stylesheet)
    original_sketch = os.path.join("../sketches_json_first_viewpoint/student9/bumps/view1_concept.json")
    original_sketch = os.path.join("../sketches_json_first_viewpoint/student8/mouse/view1_concept.json")
    orig_sketch = skio.load(original_sketch)
    match_strokes(syn_sketch, stroke_dataset)

