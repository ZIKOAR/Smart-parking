#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <WebServer.h>
#include <ArduinoJson.h>

const char* ssid = "Nom_wifi";
const char* password = "Code";
const char* websockets_server_host = "ipadress";
const uint16_t websockets_server_port = 8765;

WebServer server(80);
using namespace websockets;
WebsocketsClient client;

int carCount = 0;
StaticJsonDocument<1024> jsonBuffer;

void onMessage(WebsocketsMessage message) {
  // Affiche les données reçues sur le moniteur série
  Serial.println("Received message from server:");
  Serial.println(message.data());

  // Parser le JSON message
  // Convertir les données JSON en un objet StaticJsonDocument.
  DeserializationError error = deserializeJson(jsonBuffer, message.data());
  if (error) {
    Serial.print("deserializeJson() failed: ");
    Serial.println(error.c_str());
    return;
  }

  // Update du compteur
  carCount = jsonBuffer.size();

  // Envoyer le contenu mis à jour à la page
  server.send(200, "text/html", getHTMLContent());
}

String getHTMLContent() {
  String htmlContent = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Les Voitures dans le Parking</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      margin: 0;
      padding: 0;
      background-color: #f4f4f9;
      color: #333;
    }
    header {
      background-color: #6200ea;
      color: #fff;
      padding: 20px 0;
      text-align: center;
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    h1 {
      margin: 0;
    }
    main {
      padding: 20px;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .container {
      max-width: 800px;
      margin: auto;
      background: #fff;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .car-count {
      text-align: center;
      font-size: 24px;
      margin-bottom: 20px;
    }
    ul {
      list-style-type: none;
      padding: 0;
    }
    li {
      margin-bottom: 20px;
      padding: 15px;
      border-left: 4px solid #6200ea;
      background-color: #f0f0f0;
      border-radius: 4px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    li:last-child {
      margin-bottom: 0;
    }
  </style>
</head>
<body>
  <header>
    <h1>Parking Data</h1>
  </header>
  <main>
    <div class="container">
      <div class="car-count">Nombre des Voitures : <span id="carCount">0</span></div>
      <ul id="car_plate_list">
        <!-- List of cars will be dynamically updated -->
      </ul>
    </div>
  </main>
  <script>
    var ws = new WebSocket('ws://192.168.137.233:8765'); // WebSocket server address and port
    ws.onmessage = function(event) {
      var data = JSON.parse(event.data);
      var ul = document.getElementById('car_plate_list');
      var carCountSpan = document.getElementById('carCount');
      ul.innerHTML = '';
      carCountSpan.textContent = data.length;
      data.forEach(function(item) {
        var li = document.createElement('li');
        li.innerHTML = `
          <strong>Numero:</strong> ${item.numero}<br>
          <strong>Nb Plaque:</strong> ${item.nomplaque}<br>
          <strong>Marque:</strong> ${item.marque}<br>
          <strong>Couleur:</strong> ${item.couleur}<br>
          <strong>Date et Heure:</strong> ${item.datetime}
        `;
        ul.appendChild(li);
      });
    };

    ws.onerror = function(error) {
      console.error('WebSocket Error:', error);
    };

    ws.onclose = function() {
      console.log('WebSocket connection closed');
    };
  </script>
</body>
</html>
)rawliteral";

  return htmlContent;
}

// Fonction appelée lorsque le serveur HTTP reçoit une requête
// Envoie la page HTML via le serveur web
void handleRoot() {
  server.send(200, "text/html", getHTMLContent());
}

void setup() {
  Serial.begin(115200);

  // Connecter au WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("en cours de connexion au WiFi...");
  }
  Serial.println("WiFi connecté");

  // Démarrer le serveur HTTP
  server.on("/", HTTP_GET, handleRoot);
  server.begin();

  // Se connecter au serveur WebSocket
  client.connect(websockets_server_host, websockets_server_port, "/");
  Serial.println("Server connected");
  client.onMessage(onMessage);

  // Print IP address
  Serial.print("Adresse IP Server WebSocket: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  server.handleClient();
  if (WiFi.status() == WL_CONNECTED) {
    client.poll();
  }
}
