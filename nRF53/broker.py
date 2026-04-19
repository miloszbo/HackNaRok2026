import paho.mqtt.client as mqtt

# Konfiguracja
BROKER_IP = "127.0.0.1"  # Skrypt działa na tym samym laptopie co broker
PORT = 1883
TOPIC = "#"  # Znak '#' oznacza subskrypcję WSZYSTKICH tematów (dobre do testów)


# Funkcja wywoływana po udanym połączeniu z brokerem
def on_connect(client, userdata, flags, rc):
    print(f"Połączono z brokerem MQTT (kod błędu: {rc})")
    client.subscribe(TOPIC)
    print(f"Subskrybowanie tematu: {TOPIC} ... czekam na dane od nRF7002!")


# Funkcja wywoływana, gdy przyjdzie nowa wiadomość
def on_message(client, userdata, msg):
    print("---------------------------------")
    print(f"Temat: {msg.topic}")
    print(f"Wiadomość: {msg.payload.decode('utf-8')}")


# Inicjalizacja klienta
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Łączenie i nasłuch w nieskończonej pętli
try:
    client.connect(BROKER_IP, PORT, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\nZamykanie...")
    client.disconnect()
