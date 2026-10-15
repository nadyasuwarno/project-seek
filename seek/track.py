"""Link detections into tracks and drop short ones."""
import pandas as pd
import trackpy as tp


def search_range_px(max_speed_um_s: float, fps: float, um_per_px: float) -> float:
    return max(2.0, max_speed_um_s / fps / um_per_px)


def link(features: pd.DataFrame, search_range: float, memory: int = 3, min_len: int = 10) -> pd.DataFrame:
    if features.empty:
        return features.assign(particle=pd.Series(dtype=int))
    t = tp.link(features, search_range, memory=memory)
    t = tp.filter_stubs(t, min_len).reset_index(drop=True)
    # renumber cells 0..N-1 so IDs on screen stay small
    t["particle"] = t["particle"].astype("category").cat.codes
    return t.sort_values(["frame", "particle"]).reset_index(drop=True)
