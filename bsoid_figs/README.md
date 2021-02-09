# bsoid_figs

### Python scripts to generate manuscript figure panels

#### [fig2.py](fig2.py)
`python fig2.py` 

Runs the following subroutines
* Computes [K-fold validation accuracy](subroutines/kfold_accuracy.py), saves the accuracy_data.

`./subroutines/kfold_accuracy.py -p, path, -f, file, -o, label_order, -k, kfold_validation, -v, variable_filename`

* [Boxplot representation for K-fold validation accuracy](subroutines/accuracy_boxplot.py).

`./subroutines/accuracy_boxplot.py -p, path, -f, file, -v, variable_filename, -a, algorithm, -c, c, 
-m, fig_format, -o, outpath`

* [Plots limb trajectories](subroutines/trajectory_plot.py) for behaviors.

`./subroutines/trajectory_plot.py -p, path, -f, file, -i, animal_index, -b, bodyparts, -t, time_range,
-r, top_plot_bodyparts, -R, bottom_plot_bodyparts, -c, colors, -m, fig_format, -o, outpath`

<p align="center">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/Randomforests_Kfold_accuracy.png" width="200">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/Randomforests_frameshift_coherence.png" width="200">
</p>

Runs the following subroutines
* Computes [frameshift coherence](subroutines/frameshift_coherence.py), saves the coherence_data.

`./subroutines/frameshift_coherence.py -p, path, -f, file, -f, fps, -F, target_fps, -s, frame_skips, 
-i, animal_index, -o, label_order, -t, time_range, -v, variable_filename`

* [Boxplot representation for coherence](subroutines/coherence_boxplot.py).

`./subroutines/coherence_boxplot.py -p, path, -f, file, -v, variable_filename, -a, algorithm, -c, c, 
-m, fig_format, -o, outpath`


#### [fig5.py](fig5.py)
`python fig5.py` 



#### [fig6.py](fig6.py)
`python fig6.py` 