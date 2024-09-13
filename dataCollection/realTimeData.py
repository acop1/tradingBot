#https://pypi.org/project/websocket_client/
import websocket


def on_message(ws, message):
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    ws.send('{"type":"subscribe","symbol":"NVDA"}')

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://ws.finnhub.io?token=cpb25h1r01qniodcds9gcpb25h1r01qniodcdsa0",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close
                              )
    ws.on_open = on_open
    #take out the comment below to run the nvda stock
    #price in real time
    ws.run_forever()