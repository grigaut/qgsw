"""Configurations keys."""

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

BOX = {
    "section": "box",
    "unit": "unit",
    "x": "x",
    "y": "y",
}

IO = {
    "section": "io",
    "name": "name",
    "log performance": "performance_log",
}

SPACE = {
    "section": "space",
    "points nb x": "nx",
    "points nb y": "ny",
    "x length": "Lx",
    "y length": "Ly",
}

MODELS = {
    "section": "model",
    "section several": "models",
    "type": "type",
    "name": "name",
    "layers": "h",
    "reduced gravity": "g_prime",
    "prefix": "prefix",
    "colinearity coef": "colinearity coef",
}

OUTPUT = {
    "section": "output",
    "save": "save_results",
    "directory": "directory",
    "quantity": "quantity",
}

PERTURBATION = {
    "section": "perturbation",
    "type": "type",
    "perturbation magnitude": "perturbation_magnitude",
}

PLOTS = {
    "section": "plots",
    "show": "show",
    "save": "save",
    "directory": "plots_directory",
    "quantity": "quantity",
}

PHYSICS = {
    "section": "physics",
    "rho": "rho",
    "slip coef": "slip_coef",
    "f0": "f0",
    "beta": "beta",
    "bottom drag coefficient": "bottom_drag_coefficient",
}

SIMULATION = {
    "section": "simulation",
    "duration": "duration",
    "timestep": "dt",
    "tau": "tau",
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
