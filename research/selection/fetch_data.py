"""Download pinned public source archives; refuse truncated or changed sources."""
from pathlib import Path
import argparse,hashlib,urllib.request,os
SOURCES={
 'artists':('https://huggingface.co/datasets/yandex/yambda/resolve/main/artist_item_mapping.parquet','yambda-artist-item-mapping.parquet','cf1110aa2413ed70008ec9232d395660b2181ffda731002fb74da54e87b6f25d'),
 'kion':('https://raw.githubusercontent.com/irsafilo/KION_DATASET/main/data_original.zip','kion-data-original.zip','50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b'),
 'yambda':('https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/listens.parquet','yambda-sequential-50m-listens.parquet','c07db8326739b23fd4fb0a25037304e78b2e916320ccf1b01dc904d2b1b376c7'),
 'yambda_likes':('https://huggingface.co/datasets/yandex/yambda/resolve/main/sequential/50m/likes.parquet','yambda-sequential-50m-likes.parquet','46ade6c431a11c4a07f651b97514d0f9da5a453cb7dc5df6b2e53169597f66e0'),
}
args=argparse.ArgumentParser();args.add_argument('dataset',choices=SOURCES);a=args.parse_args()
url,name,sha=SOURCES[a.dataset]
root=Path(__file__).resolve().parents[2];dest=root/'data/raw'/name;dest.parent.mkdir(parents=True,exist_ok=True)
if not dest.exists():
 tmp=dest.with_suffix(dest.suffix+'.part')
 urllib.request.urlretrieve(url,tmp)
 with tmp.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=sha: raise SystemExit(f'Checksum mismatch; kept {tmp}. Expected {sha}, got {actual}. Do not use these bytes.')
 os.replace(tmp,dest)
with dest.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
if actual!=sha: raise SystemExit(f'Existing file checksum mismatch: {dest}')
print(dest,actual)
