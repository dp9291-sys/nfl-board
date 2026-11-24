"""Situational (schedule / travel / weather) features."""
import numpy as np
import pandas as pd

# Home stadium lat, lon, UTC offset (standard time)
STADIUM = {
    'ARI': (33.528, -112.263, -7), 'ATL': (33.755, -84.401, -5), 'BAL': (39.278, -76.623, -5),
    'BUF': (42.774, -78.787, -5), 'CAR': (35.226, -80.853, -5), 'CHI': (41.862, -87.617, -6),
    'CIN': (39.095, -84.516, -5), 'CLE': (41.506, -81.700, -5), 'DAL': (32.748, -97.093, -6),
    'DEN': (39.744, -105.020, -7), 'DET': (42.340, -83.046, -5), 'GB': (44.501, -88.062, -6),
    'HOU': (29.685, -95.411, -6), 'IND': (39.760, -86.164, -5), 'JAX': (30.324, -81.638, -5),
    'KC': (39.049, -94.484, -6), 'LA': (33.953, -118.339, -8), 'LAC': (33.953, -118.339, -8),
    'LV': (36.091, -115.184, -8), 'MIA': (25.958, -80.239, -5), 'MIN': (44.974, -93.258, -6),
    'NE': (42.091, -71.264, -5), 'NO': (29.951, -90.081, -6), 'NYG': (40.813, -74.074, -5),
    'NYJ': (40.813, -74.074, -5), 'PHI': (39.901, -75.168, -5), 'PIT': (40.447, -80.016, -5),
    'SEA': (47.595, -122.332, -8), 'SF': (37.403, -121.970, -8), 'TB': (27.976, -82.503, -5),
    'TEN': (36.166, -86.771, -6), 'WAS': (38.908, -76.864, -5),
}


def haversine_mi(a, b):
    lat1, lon1 = np.radians(a[0]), np.radians(a[1])
    lat2, lon2 = np.radians(b[0]), np.radians(b[1])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(h))


def situational(g):
    """Add rest/travel/time-zone/weather features to games frame (home-perspective)."""
    g = g.copy()
    neutral = g["location"].eq("Neutral")
    g["hfa_flag"] = (~neutral).astype(float)
    dist, tz_shift, early_west = [], [], []
    for h, a, t, n in zip(g.home_team, g.away_team, g.gametime.fillna("13:00"), neutral):
        if n:
            dist.append(0.0); tz_shift.append(0.0); early_west.append(0.0); continue
        dist.append(haversine_mi(STADIUM[a], STADIUM[h]) / 1000.0)
        sh = STADIUM[h][2] - STADIUM[a][2]  # +3 => west team flying east
        tz_shift.append(sh)
        hr = int(str(t).split(":")[0])
        # body-clock: west-coast team playing a 1pm ET kickoff (10am at home)
        early_west.append(1.0 if (sh >= 2 and hr < 14) else 0.0)
    g["away_travel_k"] = dist
    g["tz_shift"] = tz_shift
    g["early_west"] = early_west
    # west team travelling east for a night kickoff (tested; not used in the model)
    g["night_west_east"] = [1.0 if (s >= 2 and int(str(t).split(":")[0]) >= 19) else 0.0
                           for s, t in zip(g.tz_shift, g.gametime.fillna("13:00"))]
    hr_, ar_ = g["home_rest"].clip(4, 14), g["away_rest"].clip(4, 14)
    g["rest_diff"] = (hr_ - ar_) / 7.0
    g["home_bye"] = (g["home_rest"] >= 12).astype(float)
    g["away_bye"] = (g["away_rest"] >= 12).astype(float)
    g["home_short"] = (g["home_rest"] <= 5).astype(float)
    g["away_short"] = (g["away_rest"] <= 5).astype(float)
    indoor = g["roof"].isin(["dome", "closed"])
    g["wind_hi"] = ((~indoor) & (g["wind"].fillna(0) >= 15)).astype(float)
    g["cold"] = ((~indoor) & (g["temp"].fillna(60) <= 32)).astype(float)
    g["div"] = g["div_game"].fillna(0).astype(float)
    return g
