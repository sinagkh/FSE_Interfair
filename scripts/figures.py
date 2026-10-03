"""Render paper figures from numeric exports without querying or training models."""
from pathlib import Path
import argparse,shutil
import figure_source as f
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=f.ROOT/'reproduced/figures');a=p.parse_args()
f.OUT=a.out.resolve();f.OUT.mkdir(parents=True,exist_ok=True)
shutil.copyfile(f.ROOT/'results/figures/response_example.json',f.OUT/'response_example.json')
f.fig_example();f.fig_maintenance();f.fig_heatmap()
print('Figures reproduced in',f.OUT)
