import pandas as pd
import numpy as np
from coco_names import coco_category_names as coco_names


def get_coco_name(i):
	return coco_names[i]


distance_limit = 20

df = pd.read_csv("output.txt")
df['timestamp'] = df['timestamp'].apply(np.floor).astype(int)
df = df.loc[df['min_distance'] <= distance_limit]
o = df.groupby(by=['timestamp', 'frame_id', 'id'])['id_type'].count().reset_index()
o.columns = ['timestamp', 'frame', 'id', 'count']
o = o.groupby(by=['timestamp', 'id'])['count'].mean().reset_index()
o['count'] = o['count'].round(0).astype(int)
o['class'] = o['id'].apply(get_coco_name)
o = o.rename(columns={'count': ''})
x = pd.pivot_table(o, index=['timestamp'], columns=['class'], values=[''], fill_value=0, dropna=True).reset_index()
x.columns = x.columns.droplevel(0)
x = x.rename(columns={'': 'timestamp'})
x.to_csv('output_aggregated.csv', index=False)
