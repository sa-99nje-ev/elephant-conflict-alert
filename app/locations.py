# This file stores coordinates for our key locations
# (Latitude, Longitude)

LOCATIONS = {
    "Anuradhapura": (8.3114, 80.4037),
    "Polonnaruwa": (7.9403, 81.0188),
    "Ampara": (7.2947, 81.6748),
    "Monaragala": (6.8724, 81.3496),
    "Puttalam": (8.0343, 79.8430),
    "Hambantota": (6.1240, 81.1185)
}

def get_coords(location_name: str):
    """
    Gets coordinates for a given location name.
    Returns a (lat, lon) tuple or None if not found.
    """
    return LOCATIONS.get(location_name)

def get_location_names():
    """
    Returns a list of all location names.
    """
    return list(LOCATIONS.keys())