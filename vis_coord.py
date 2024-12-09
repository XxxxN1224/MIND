import pandas as pd
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path

id = '3df7a6c0-5f19-490f-89df-931cfdb2e503'
raw = f'/home/fqf/fqf_folder/01_Git/MIND/data/{id}/scenario_{id}.parquet'
map_path = f'/home/fqf/fqf_folder/01_Git/MIND/data/{id}/log_map_archive_{id}.json'

static_map_path = Path(map_path)
df = pd.read_parquet(raw)
static_map = ArgoverseStaticMap.from_json(static_map_path)

# 过滤 track_id 为 "AV" 的行
av_positions = df[df['track_id'] == 'AV']

# 假设有 timestamp 列，我们根据 timestamp 找到最后一个位置
if 'timestamp' in df.columns:
    last_av_position = av_positions.loc[av_positions['timestamp'].idxmax()]
else:
    # 如果没有 timestamp 列，则直接取最后一行
    last_av_position = av_positions.iloc[-1]

# 获取最后一个 position_x 和 position_y
last_position_x = last_av_position['position_x']
last_position_y = last_av_position['position_y']

# 输出结果
print(f"Last position_x for track_id 'AV': {last_position_x}")
print(f"Last position_y for track_id 'AV': {last_position_y}")