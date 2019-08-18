import json

def tryCastToFloat(str):
  # Cast an item to a float if it exists.
  try:
    return float(str)
  except:
    return None

def tryCastToInt(str):
  # Cast an item to an int if it exists.
  try:
    return int(str)
  except:
    return None

def cat_to_name(jsonFilepath):
    with open(jsonFilepath, 'r') as f:
        return json.load(f)