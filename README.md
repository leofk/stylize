# CAD2Sketch

This is the official repository for the publication [CAD2Sketch: Generating Concept Sketches from CAD Sequences](https://ns.inria.fr/d3/cad2sketch/).

If you should encounter any problems with the code, don't hesitate to contact me
at felix.hahnlein@inria.fr.

# Installation

    git clone https://gitlab.inria.fr/D3/cad2sketch.git
    cd cad2sketch
    conda env create -f environment.yml
    conda activate cad2sketch

## Dependencies

- [SynDraw](https://gitlab.inria.fr/D3/contour-detect)
- [Gurobi](https://www.gurobi.com/)

# Script usage

The program follows the pipeline described in the paper.

Based on an [Onshape](https://www.onshape.com/en/) document, we **download** the 
data of the intermediate steps of the CAD sequence.
Then, we **generate** construction and feature lines.
Next, **line selection** is performed via a discrete optimization problem.
Finally, we **stylize** lines to give them a real-world aesthetic.

## Download
To download information from Onshape, you need to open an account. Once you are logged in, you can [obtain an API Key](https://dev-portal.onshape.com/keys) which let's you communicate with Onshape via the [REST API](https://onshape-public.github.io/docs/apioverview/).
Copy-paste your *access key* and your *secret key* into the corresponding fields in <code>onshape/creds/creds.json</code>.

All the data related to an Onshape document will be stored in a single folder in the <code>data</code> folder. After creating this folder, we can download the document:

    mkdir data/tmp_doc
    python single_url_processing.py --data_folder=data/tmp_doc --collect_data=true --url=your_onshape_document_url

Note: we automatically copy the entire Onshape workspace into your own documents. It is a good idea to delete this document if not needed anymore to alleviate the servers.

## Line generation
In the file <code>single_url_processing.py</code>, change the variable <code>SYN_DRAW_PATH</code> to where you installed SynDraw.

At this point, we need to choose a viewpoint from where we want to produce a sketch.
The viewpoint is specified using [spherical coordinates](https://en.wikipedia.org/wiki/Spherical_coordinate_system), centered around the bounding box center of the object, so we have to specify a theta and phi angle and a radius.
The radius is a multiplier of the bounding box diagonal, so a radius of <code>1.0</code> places the camera one bounding box diagonal away from the bounding box center.
Empirically, we found that a radius of <code>1.4</code> produces nice results.

    python single_url_processing.py --data_folder=data/tmp_doc --generate_silhouette_lines=true --recompute_all_construction_lines=true --theta=your_theta_angle --phi=your_phi_angle --radius=your_radius

All view-dependent data will be stored in the subfolder <code>data/tmp_doc/your_theta_your_phi_your_radius</code>.

## Line selection
## Line stylization


# Code structure
