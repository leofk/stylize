batch="teaser"
file_name="Professional1_tubes_view1_340_depth_fixed_regularized"
designer=Professional1
stylesheet_name=Professional1

python single_url_processing.py \
    --dir_1 ${batch}/teaser \
    --dir_2 ${batch}/teaser \
    --file_1 ${file_1} \
    --file_2 ${file_2} \
    --stylesheet_name ${stylesheet_name} \
    --designer ${designer}
