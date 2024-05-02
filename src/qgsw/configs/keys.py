"""Configurations keys."""

PHYSICS = {
    "section": "physics",
    "rho": "rho",
    "slip coef": "slip_coef",
    "f0": "f0",
    "beta": "beta",
    "bottom drag coefficient": "bottom_drag_coefficient",
}

LAYERS = {
    "section": "layers",
    "layer thickness": "h",
    "reduced gravity": "g_prime",
}

MESH = {
    "section": "mesh",
    "points nb x": "nx",
    "points nb y": "ny",
    "x length": "Lx",
    "y length": "Ly",
    "timestep": "dt",
}

BOX = {
    "section": "box",
    "unit": "unit",
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
    "log performance": "performance_log",
}

OUTPUT = {
    "section": "output",
    "save": "save_results",
    "directory": "directory",
    "quantity": "quantity",
}

PLOTS = {
    "section": "plots",
    "show": "show",
    "save": "save",
    "directory": "plots_directory",
    "quantity": "quantity",
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

VORTEX = {
    "section": "vortex",
    "type": "type",
    "perturbation magnitude": "perturbation_magnitude",
}
