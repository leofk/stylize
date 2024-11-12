import numpy as np


def optimize_opacities(edges, stylesheet):

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
        #if tmp_max_feature_id == max_feature_id and "silhouette_line" in all_line_types:
            style_labels.append("silhouette_line")
            continue
        if tmp_max_feature_id == max_feature_id and ("feature_line" in all_line_types or edge["type"] == "feature_line"):
        #if tmp_max_feature_id == max_feature_id and "feature_line" in all_line_types:
            style_labels.append("feature_line")
            continue
        style_labels.append("grid_lines")

    print("Style Labels")
    for i, s in enumerate(style_labels):
        print(i, s)

    old_opacities = []
    for edge_id, s_l in enumerate(style_labels):
        mu = None
        sigma = None
        if s_l == "grid_lines":
            mu = stylesheet["opacities_per_type"]["scaffold"]["mu"]
            sigma = stylesheet["opacities_per_type"]["scaffold"]["sigma"]
        elif s_l == "feature_line":
            if edges[str(edge_id)]["visibility_score"] > 0.5:
                mu = stylesheet["opacities_per_type"]["vis_edges"]["mu"]
                sigma = stylesheet["opacities_per_type"]["vis_edges"]["sigma"]
            else:
                mu = stylesheet["opacities_per_type"]["occ_edges"]["mu"]
                sigma = stylesheet["opacities_per_type"]["occ_edges"]["sigma"]
        elif s_l == "silhouette_line":
            mu = stylesheet["opacities_per_type"]["silhouette"]["mu"]
            sigma = stylesheet["opacities_per_type"]["silhouette"]["sigma"]
        #sampled_op = min(1.0, max(0.0, np.random.normal(loc=mu, scale=sigma/3, size=1)[0]))
        #sampled_op = mu
        sampled_op = min(1.0, max(0.0, np.random.normal(loc=mu, scale=sigma/5, size=1)[0]))
        #if s_l == "grid_lines":
        #    sampled_op = 0.05
        old_opacities.append(sampled_op)
    return old_opacities
