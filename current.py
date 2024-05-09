import geocoder
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my_geocoder")


def get_city_name(latitude, longitude):
    location = geolocator.reverse((latitude, longitude))
    if location and 'address' in location.raw:
        address = location.raw['address']
        city = address.get('city', '')
        if not city:
            city = address.get('town', '')
        if not city:
            city = address.get('village', '')
        return city
    else:
        return None


geolocator = Nominatim(user_agent="my_geocoder")


def get_live_location():
    location = geocoder.ip('me')
    if location:
        return location.latlng
    else:
        return None


def get_city_name(latitude, longitude):
    location = geolocator.reverse((latitude, longitude))
    if location and 'address' in location.raw:
        address = location.raw['address']
        city = address.get('city', '')
        if not city:
            city = address.get('town', '')
        if not city:
            city = address.get('village', '')
        return city
    else:
        return None


latitude, longitude = get_live_location()

if latitude is not None and longitude is not None:
    print("Latitude:", latitude)
    print("Longitude:", longitude)

    city = get_city_name(latitude, longitude)
    if city:
        print("City Name:", city)
    else:
        print("Failed to retrieve city name.")
else:
    print("Failed to retrieve live location.")
