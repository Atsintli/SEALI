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
  
  clase_0 = data[0][0]
  clase_1 = data[0][1]
  clase_2 = data[0][2]
  clase_3 = data[0][3]
  clase_4 = data[0][4]
  clase_5 = data[0][5]
  clase_6 = data[0][6]
  # clase_7 = data[0][7]
  # clase_8 = data[0][8]
  # clase_9 = data[0][9]

  event = max([clase_0,clase_1,clase_2,clase_3,clase_4,clase_5,clase_6
  #,clase_7,clase_8,clase_9
  ])

  if event == clase_0:
    print (event, "\t", "clase_0")
  if event == clase_1:
    print (event, "\t", "clase_1")
  if event == clase_2:
    print (event, "\t", "clase_2")
  if event == clase_3:
    print (event, "\t", "clase_3")
  if event == clase_4:
    print (event, "\t", "clase_4")
  if event == clase_5:
    print (event, "\t", "clase_5")
  if event == clase_6:
     print (event, "\t", "clase_6")

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