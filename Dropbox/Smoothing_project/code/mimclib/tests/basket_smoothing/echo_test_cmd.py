#!/usr/bin/python
import numpy as np
import argparse # this module to add arguments

# here we add the arguments that we can call

parser = argparse.ArgumentParser(add_help=True)
parser.register('type', 'bool',
                lambda v: v.lower() in ("yes", "true", "t", "1"))
parser.add_argument("-db", type="bool", action="store", default=False)
parser.add_argument("-db_tag", type=str, action="store",
                    default="basket_001", help="Database tag")
parser.add_argument("-qoi_dim", type=int, action="store",
                    default=0, help="MISC dim")   
parser.add_argument("-tries", type=int, action="store",
                          default=0, help="Number of realizations") 

args, unknowns = parser.parse_known_args()

#dim = args.qoi_dim

dim = args.qoi_dim



if dim == 0:
  base = "mimc_run.py -mimc_TOL {TOL} -mimc_min_dim {the_dim} -qoi_dim {the_dim} -ksp_rtol 1e-25 -ksp_type gmres -mimc_M0 1 \
  -mimc_beta {beta} -mimc_gamma {gamma} -mimc_h0inv {h0inv} \
  -mimc_bayes_fit_lvls 3 -mimc_moments 1 -mimc_bayesian False".format(bayesian="{bayesian}", TOL="{TOL}", the_dim=dim, seed="{seed}",
             h0inv="2",  gamma="2.8",  beta="2")


# In case of 1 D problem
# elif dim == 1:
#   base = "mimc_run.py -mimc_TOL {TOL} -mimc_min_dim {the_dim} -qoi_dim {the_dim} -ksp_rtol 1e-25 -ksp_type gmres -mimc_M0 1 \
# -mimc_beta {beta} -mimc_gamma {gamma} -mimc_h0inv {h0inv} \
# -mimc_bayes_fit_lvls 3 -mimc_moments 1 -mimc_bayesian False".format(bayesian="{bayesian}", TOL="{TOL}", the_dim=dim, seed="{seed}",
#            h0inv="2",
#                                    gamma="2.8",
#                                    beta="2")



base += " ".join(unknowns)

if not args.db:
    cmd_single = "python " + base + " -mimc_verbose 10 -db True -db_tag basket_001 " #verbose >0 detailed printing results
    print(cmd_single.format(TOL=0.1))
else:
    #cmd_multi = "python " + base + " -mimc_verbose 0 -db True -db_tag test_001 "
    #print cmd_multi.format(tag=args.db_tag.format(dim), TOL=1e-10)
    cmd_multi = "python " + base + " -mimc_verbose 10 -db True -db_tag basket_6d"
    for i in range(0, args.tries):
       print cmd_multi.format(tag=args.db_tag.format(dim), TOL=0.0001)
