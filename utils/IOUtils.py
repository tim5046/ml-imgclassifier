def yesOrNo(question):
  while True:
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[:1] == 'y':
      return True
    if reply[:1] == 'n':
      return False

def getResponse(question, allowableResponses=None):
  while True:
    reply = str(input(question+': ')).lower().strip()
    if len(reply) > 0:
      if allowableResponses:
        # Ensure response is in the list of allowableResponses
        if reply in allowableResponses:
          return reply
      else:
        # No validation, just return whatever they give us
        return reply

def notify(message):
  print(f"\n\n#######\n{message}")