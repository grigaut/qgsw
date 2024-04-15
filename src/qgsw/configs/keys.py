"""Configurations keys."""

PHYSICS = {
    "section": "physics",
    "rho": "rho",
    "slip coef": "slip_coef",
    "f0": "f0",
    "beta": "beta",
    "wind stress magnitude": "wind_stress_mag",
    "drag coefficient": "drag_coefficient",
}

LAYERS = {
    "section": "layers",
    "layer thickness": "h",
    "reduced gravity": "g_prime",
}

GRID = {
    "section": "grid",
    "box unit": "box_unit",
    "points nb x": "nx",
    "points nb y": "ny",
    "x length": "Lx",
    "y length": "Ly",
    "timestep": "dt",
}

BOX = {
    "section": "box",
    "x": "x",
    "y": "y",
}

BATHY = {
    "section": "bathymetry",
    "h top ocean": "htop_ocean",
    "island minimum area": "island_min_area",
    "lake minimum area": "lake_min_area",
    "interpolation": "interpolation",
}

BATHY_DATA = {
    "section": "data",
    "url": "URL",
    "folder": "folder",
    "longitude": "longitude",
    "latitude": "latitude",
    "elevation": "elevation",
}

IO = {
    "section": "io",
    "name": "name",
    "output directory": "output_dir",
    "log performance": "performance_log",
    "plot frequency": "plot_frequency",
}

WINDSTRESS = {
    "section": "windstress",
    "magnitude": "magnitude",
    "type": "type",
    "drag coefficient": "drag_coefficient",
}

WINDSTRESS_DATA = {
    "section": "data",
    "url": "URL",
    "folder": "folder",
    "data": "data",
    "longitude": "longitude",
    "latitude": "latitude",
    "time": "time",
    "field 1": "field_1",
    "field 2": "field_2",
    "method": "method",
}
