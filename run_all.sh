#!/bin/bash

# weibull
#alph = 0.98
bash tce_graph.sh dist=weibull p1=0.6 p2=1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=0.8 p2=1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=1.0 p2=1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=1.2 p2=1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
#alph = 0.99
bash tce_graph.sh dist=weibull p1=0.6 p2=1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=0.8 p2=1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=1.0 p2=1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=1.2 p2=1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
#alph = 0.999
bash tce_graph.sh dist=weibull p1=0.6 p2=1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=0.8 p2=1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=1.0 p2=1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=weibull p1=1.2 p2=1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop

# lognormal
#alph = 0.98
bash tce_graph.sh dist=lnorm p1=0 p2=0.1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.3 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.5 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.7 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
#alph = 0.99
bash tce_graph.sh dist=lnorm p1=0 p2=0.1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.3 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.5 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.7 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
#alph = 0.999
bash tce_graph.sh dist=lnorm p1=0 p2=0.1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.3 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.5 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=lnorm p1=0 p2=0.7 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop

# gpd
#alph = 0.98
bash tce_graph.sh dist=gpd p1=0.3 p2=1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.5 p2=1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.7 p2=1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.9 p2=1 alph=0.98 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
#alph = 0.99
bash tce_graph.sh dist=gpd p1=0.3 p2=1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.5 p2=1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.7 p2=1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.9 p2=1 alph=0.99 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
#alph = 0.999
bash tce_graph.sh dist=gpd p1=0.3 p2=1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.5 p2=1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.7 p2=1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop
bash tce_graph.sh dist=gpd p1=0.9 p2=1 alph=0.999 tp_select=search tp_init=0.9 tp_num=100 signif=0.1 cutoff=0.9 stop_rule=forward_stop