#This program listens to several addresses, and prints some information about received packets.

import argparse
import math
import requests # importing the requests library 
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import json

client = udp_client.SimpleUDPClient('127.0.0.1', 5006) #this client sends to SC

def tf_handler(unused_addr, *args):
  # sending get request and saving the response as response object 
  headers = {"content-type": "application/json"}
  data = {"instances": [[*args]]}
  r = requests.post(url = "http://localhost:8501/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  # extracting data in json format 
  data = r.json()["predictions"]
  client.send_message("/clase", *data)
  
  #caotico  = data[0][0]
  complejo = data[0][0]
  fijo = data[0][1]
  periodico = data[0][2]

  file = open('clases_database.txt', 'a')

  printable_data = "Compuesto"
  if complejo > 0.333333333333:
    printable_data = "Complejo"
  if fijo > 0.333333333333:
    printable_data = "Fijo"
  if periodico > 0.333333333333:
    printable_data = "Periodico"
  print(printable_data)
  #print(data)

  file.write(printable_data  + ", ")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on") 
  parser.add_argument("--port",
      type=int, default=5005, help="The port to listen on") 
  args = parser.parse_args()

  dispatcher = dispatcher.Dispatcher()
  dispatcher.map("/features", tf_handler)
 
  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()

  file.close()
