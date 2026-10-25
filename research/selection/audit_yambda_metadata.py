from pathlib import Path
import polars as pl,json,hashlib
root=Path(__file__).resolve().parents[2]
p=root/'data/raw/yambda-artist-item-mapping.parquet'
seq=pl.read_parquet(root/'data/processed/yambda-1000-users-full-histories.parquet')
ids=seq.select(pl.col('item_id').explode()).unique()['item_id']
q=pl.scan_parquet(p)
mp=q.filter(pl.col('item_id').is_in(ids.implode())).collect()
mp.write_parquet(root/'data/processed/yambda-sampled-item-artists.parquet')
per=mp.group_by('item_id').agg(pl.col('artist_id').n_unique().alias('n_artists'))
out={'source':'https://huggingface.co/datasets/yandex/yambda/resolve/main/artist_item_mapping.parquet','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'file_bytes':p.stat().st_size,'source_rows':q.select(pl.len()).collect().item(),'sample_distinct_items':len(ids),'mapped_sample_items':len(per),'sample_item_coverage':len(per)/len(ids),'sample_items_with_multiple_artists':int((per['n_artists']>1).sum()),'max_artists_per_sample_item':per['n_artists'].max(),'note':'Static catalog relation assumption must be stated. For many-to-many features, deduplicate event-artist joins; do not multiply raw user counts by collaborator count.'}
(root/'research/selection/yambda-artist-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
