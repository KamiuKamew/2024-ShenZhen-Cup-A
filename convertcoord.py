def convert(geodetic_coords) -> list[(float, float, float, float)]:
    """
    Convert geodetic coordinates to Cartesian coordinates.
    :param geodetic_coords: a list of tuples of (longitude, latitude, height, record_time) in degrees, meters, and seconds.
    :return: a list of tuples of (x, y, z, t) in meters and seconds.
    """
    
    longitude_to_meter = 97304
    latitude_to_meter = 111263
    height_to_meter = 1

    origin = geodetic_coords[0]
    results = []
    for coord in geodetic_coords:
        x = (coord[0] - origin[0]) * longitude_to_meter
        y = (coord[1] - origin[1]) * latitude_to_meter
        z = (coord[2] - origin[2]) * height_to_meter
        t = (coord[3] - origin[3])
        results.append((x, y, z, t))
    
    return results