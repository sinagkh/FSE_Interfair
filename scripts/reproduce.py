"""Rebuild primary and transfer tables from the supplied per-seed results."""
from pathlib import Path
import argparse, json
import pandas as pd
import table_format as tf
ROOT = Path(__file__).resolve().parents[1]
ID = ['task', 'architecture', 'arm', 'seed']

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'reproduced');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    primary=pd.read_csv(ROOT/'results/main/paper_primary_per_seed.csv')
    assert len(primary)==800 and not primary.duplicated(ID).any()
    for key,g in primary.groupby(ID[:-1]):assert sorted(g.seed)==list(range(1000,1010)),key
    summary=primary.groupby(ID[:-1])[tf.METRICS].agg(['mean','std'])
    summary.columns=[f'{m}_{s}' for m,s in summary.columns];summary.reset_index().to_csv(out/'primary_mean_sd.csv',index=False)
    ranks=pd.concat([pd.read_csv(ROOT/f'results/main/{p}_sk_ranks.csv') for p in ['primary','states']],ignore_index=True)
    tables=out/'tables';tables.mkdir(exist_ok=True)
    for (task,arch),g in primary.groupby(['task','architecture']):
        rows=tf.block_rows(primary,ranks,task,arch)
        (tables/f'{task}_{arch}.tex').write_text('% AUROC, accuracy, F1, AOD, DP, EO, residual, V (%), D (%), W (%)\n'+'\n'.join(rows)+'\n')
    transfer=pd.read_csv(ROOT/'results/main/paper_transfer_per_seed.csv');assert len(transfer)==320 and not transfer.duplicated(ID).any()
    trank=pd.read_csv(ROOT/'results/main/transfer_sk_ranks.csv')
    for (task,arch),g in transfer.groupby(['task','architecture']):
        (tables/f'transfer_{task}_{arch}.tex').write_text('\n'.join(tf.block_rows(transfer,trank,task,arch))+'\n')
    stats={'primary_runs':len(primary),'primary_settings':primary[['task','architecture']].drop_duplicates().shape[0],'transfer_runs':len(transfer),'seed_values':sorted(primary.seed.unique().tolist()),'primary_means_and_tables':'PASS'}
    (out/'checks.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
