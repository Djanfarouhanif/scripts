from openlocationcode import openlocationcode

latitude = 6.1319
longitude = 1.2228

def genaretePlusCode(latitude, longitude):

    plus_code = openlocationcode.encode(latitude,longitude)

    return plus_code


def decodePlusCode(plusCode):
    decoded = openlocationcode.decode(plusCode)
    latitude = decoded.latitudeCenter
    longitude = decoded.longitudeCenter

    return latitude, longitude
plusCode = genaretePlusCode(latitude,longitude)

latitude, longitude = decodePlusCode(plusCode)

print(plusCode, latitude, longitude)