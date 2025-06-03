#!/bin/bash

# source ../.env

export JAMUN_PLOT_PATH=plots

python make_plots.py --experiment Cremp_4AA_5AA --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN
python make_plots.py --experiment Cremp_4AA_long_train --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN
#python make_plots.py --experiment Our_2AA --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN
#python make_plots.py --experiment Timewarp_2AA --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN
#python make_plots.py --experiment Timewarp_4AA --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN
#python make_plots.py --experiment MDGen_4AA --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN
#python make_plots.py --experiment Our_5AA --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN
#python make_plots.py --experiment Timewarp_4AA_0.4A --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN
#python make_plots.py --experiment Chignolin --plot-dir ${JAMUN_PLOT_PATH} --trajectory JAMUN_2x
