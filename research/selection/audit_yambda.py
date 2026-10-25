from pathlib import Path
import polars as pl
import random,json,hashlib,time
root=Path(__file__).resolve().parents[2]
p=root/'data/raw/yambda-sequential-50m-listens.parquet'
q=pl.scan_parquet(p)
lengths=q.select('uid',pl.col('item_id').list.len().alias('n'),pl.col('timestamp').list.min().alias('min_t'),pl.col('timestamp').list.max().alias('max_t')).collect()
assert lengths['uid'].n_unique()==len(lengths), 'Expected one sequence per user'
rng=random.Random(42)
uids=rng.sample(sorted(lengths['uid'].to_list()),1000)
sample=q.filter(pl.col('uid').is_in(uids)).collect()
(root/'data/processed').mkdir(exist_ok=True,parents=True)
sample.write_parquet(root/'data/processed/yambda-1000-users-full-histories.parquet')
lens=sample.select([pl.col(c).list.len().alias(c) for c in sample.columns if c!='uid'])
assert all((lens[c]==lens['item_id']).all() for c in lens.columns), 'Misaligned sequence lists'
flat=sample.explode([c for c in sample.columns if c!='uid'])
# This audit is descriptive; no ranking/reward test score is evaluated.
summary=flat.select(pl.len().alias('events'),pl.col('item_id').n_unique().alias('items'),pl.col('timestamp').min().alias('min_t'),pl.col('timestamp').max().alias('max_t'),pl.col('played_ratio_pct').min().alias('ratio_min'),pl.col('played_ratio_pct').max().alias('ratio_max'),pl.col('played_ratio_pct').null_count().alias('ratio_null'),pl.col('track_length_seconds').null_count().alias('length_null'),pl.col('track_length_seconds').median().alias('track_length_median'),(pl.col('is_organic')==1).mean().alias('organic_fraction'),(pl.col('played_ratio_pct')>100).sum().alias('ratio_above_100'),(pl.col('played_ratio_pct')==0).sum().alias('ratio_zero')).to_dicts()[0]
order=sample.select(pl.col('timestamp').list.eval(pl.element().diff().drop_nulls()<0).list.any().sum().alias('histories_with_decreasing_time'),pl.col('timestamp').list.eval(pl.element().diff().drop_nulls()==0).list.sum().sum().alias('equal_time_adjacent_pairs')).to_dicts()[0]
assert order['histories_with_decreasing_time']==0, 'Unsorted times'
# Train-only vocabulary coverage: choose a provisional cutoff from time range,
# not by inspecting held-out performance. This is a feasibility check, not final split.
min_t=summary['min_t']; max_t=summary['max_t']; cutoff=min_t+int((max_t-min_t)*0.7)
train=flat.filter(pl.col('timestamp')<cutoff)
counts=train.group_by('item_id').len().sort('len',descending=True)
coverage={}
for k in [20000,50000]:
 ids=counts.head(k)['item_id']
 coverage[str(k)]={'actual_vocab':len(ids),'train_event_fraction':train['item_id'].is_in(ids.implode()).mean(),'note':'Train-side only. No validation/test target coverage or quality used for selection here.'}
report={'source':'https://huggingface.co/datasets/yandex/yambda','file':'sequential/50m/listens.parquet','size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'schema':{k:str(v) for k,v in q.collect_schema().items()},'all_listen_users':len(lengths),'all_listen_events':int(lengths['n'].sum()),'history_quantiles':{str(x):lengths['n'].quantile(x) for x in [0.5,0.75,0.9,0.99]},'sample_seed':42,'sample_users':1000,'complete_sampled_user_histories':True,'sample':summary,'sequence_checks':{'lists_aligned':True,**order},'provisional_train_cutoff_native_timestamp':cutoff,'train_events':len(train),'train_vocab_coverage':coverage,'timestamp_note':'Paper section3.3 establishes seconds from a common undisclosed origin, rounded down to multiples of5. Do not multiply by5. Equal timestamps cannot be ordered with hindsight.','warnings':['Actual parquet columns use singular item_id and timestamp; README sequential examples use plural names.','Released listens file has fewer than10,000 users; count actual rows rather than claiming every advertised user has listens.','No impression logging probabilities: predicted reward is observational and not causal long-term benefit.']}
(root/'research/selection/yambda-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
