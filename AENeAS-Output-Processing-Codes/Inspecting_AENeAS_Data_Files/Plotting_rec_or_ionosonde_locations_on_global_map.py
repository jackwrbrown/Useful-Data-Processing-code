# This is an easy code to plot the locations of certain receivers/ionosondes on a flat Earth map

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import aacgmv2
from datetime import datetime

# === Settings ===
time4mag = datetime(2025, 6, 2)
alt4mag = 300  # km altitude for magnetic conversion

# === Station lists ===
list1 = [
    {"name": "ALPENA", "lat": 45.07, "lon": -83.56},
    {"name": "EIELSON", "lat": 64.66, "lon": -147.07},
    {"name": "FAIRFORD", "lat": 51.70, "lon": -1.50},
    {"name": "IDAHO", "lat": 43.81, "lon": -112.68},
    {"name": "JULIUSRUH", "lat": 54.60, "lon": 13.40},
    {"name": "MOSCOW", "lat": 55.49, "lon": 37.29},
    {"name": "TROMSO", "lat": 69.60, "lon": 19.20},
]

list2 = [
    {"name": "Eglin AFB", "lat": 30.50, "lon": -86.5},
    {"name": "El Arenosillo", "lat": 37.1, "lon": -6.7},
    {"name": "Fairford", "lat": 51.70, "lon": -1.50},
    {"name": "Ascension Island", "lat": -8.0, "lon": -14.4},
    {"name": "Wake", "lat": 19.3, "lon": 166.6},
    {"name": "Grahamstown", "lat": -33.3, "lon": 26.5},
    
]

# Normalize and title-case names
normalize = lambda name: name.strip().lower()
titlecase = lambda name: name.strip().capitalize() if name.isupper() else name.strip().title()

# Build dicts
dict1 = {normalize(s["name"]): {**s, "name": titlecase(s["name"])} for s in list1}
dict2 = {normalize(s["name"]): {**s, "name": titlecase(s["name"])} for s in list2}

both = {k: dict1[k] for k in dict1 if k in dict2}
only1 = {k: dict1[k] for k in dict1 if k not in dict2}
only2 = {k: dict2[k] for k in dict2 if k not in dict1}

# === Custom label offsets for each station group ===
offsets_only1 = {
    "alpena": (-12, -6),
    "eielson": (4, 0),
    "fairford": (-6, -10),
    "idaho": (-9, -6),
    "juliusruh": (-9, -6),
    "moscow": (-9, -6),
    "tromso": (-12, 3),
}

offsets_only2 = {
    "eglin afb": (4, -1),
    "el arenosillo": (-12, -6),
    "ascension island": (-12, -6),
    "wake": (-12, -6),
    "grahamstown": (-12, -6),
    "fairford": (-5, -5),
}

offsets_both = {
    "fairford": (-15, -6),
}

# === Function to draw geomagnetic latitude lines ===
def draw_magnetic_latitude_line(ax, mag_lat_target, time4mag, alt4mag, **plot_kwargs):
    lons = np.linspace(-180, 180, 361)
    geo_lats = []
    geo_lons = []
    for lon in lons:
        lat_geo, lon_geo, _ = aacgmv2.convert_latlon(mag_lat_target, lon, alt4mag, time4mag, method_code='A2G')
        geo_lats.append(lat_geo)
        geo_lons.append(lon_geo)

    # Unwrap longitudes to avoid line breaks at ±180°
    geo_lons = np.unwrap(np.radians(geo_lons))
    geo_lons = np.degrees(geo_lons)

    ax.plot(geo_lons, geo_lats, transform=ccrs.PlateCarree(), **plot_kwargs)

# === Create map ===
fig = plt.figure(figsize=(15, 7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

ax.add_feature(cfeature.LAND, facecolor='lightgreen')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
ax.coastlines()

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
gl.xlabel_style = {'size': 14, 'weight': 'bold', 'color': 'black'}
gl.ylabel_style = {'size': 14, 'weight': 'bold', 'color': 'black'}

# === Draw geomagnetic latitude lines ±60° ===
draw_magnetic_latitude_line(ax, 60, time4mag, alt4mag,
                           color='black', linestyle='--', linewidth=2, label='Mag Lat ±60°')
draw_magnetic_latitude_line(ax, -60, time4mag, alt4mag,
                           color='black', linestyle='--', linewidth=2)


# === Plot stations with labels (bold, custom offsets) ===
def plot_stations(stations, color, label, offsets):
    for s in stations.values():
        # Draw outline (black bigger star)
        ax.plot(s["lon"], s["lat"], marker='*', markersize=18, markeredgewidth=2,
                markeredgecolor='black', markerfacecolor='none', transform=ccrs.Geodetic())
        
        # Draw main star (colored smaller star)
        ax.plot(s["lon"], s["lat"], marker='*', color=color, markersize=12,
        transform=ccrs.Geodetic(), label=label)
        name_key = s["name"].lower()
        dx, dy = offsets.get(name_key, (-12, -6))  # default offset
        ax.text(s["lon"] + dx, s["lat"] + dy, s["name"], fontsize=11, fontweight='bold',
                transform=ccrs.Geodetic())
        label = None  # Only label first point to avoid duplicate legend entries
plot_stations(only2, 'red', 'Optimisation Ionosonde Sites', offsets_only2)
plot_stations(only1, 'blue', 'Validation Ionosonde Sites', offsets_only1)

plot_stations(both, 'darkgreen', 'Optimisation & Validation Sites', offsets_both)

# === Legend & title ===
plt.legend(loc='lower left', fontsize=14)

plt.tight_layout()
plt.show()
