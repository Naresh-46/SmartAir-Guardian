// ============================================================
//  SmartAir Guardian — HTTP Client Firmware (LIVE PREDICTIONS)
//  ESP8266 NodeMCU + CD4051 + MQ135/2/7/4/3 + DHT22
//  Sends sensor data to Flask server → receives ML predictions
// ============================================================

#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>

// ── WiFi Configuration ──────────────────────────────────────
const char* WIFI_SSID     = "Motorola Edge 50 Fusion";  // Your WiFi SSID
const char* WIFI_PASSWORD = "12345678";                 // Your WiFi password
const char* SERVER_IP     = "172.17.212.86";            // Your laptop IP
const int   SERVER_PORT   = 5000;
const char* API_ENDPOINT  = "/api/predict";

// ── Pin Definitions ─────────────────────────────────────────
#define S0_PIN     D3
#define S1_PIN     D4
#define S2_PIN     D5
#define DHT_PIN    D0
#define FLAME_PIN  D6
#define LED_GREEN  D7
#define LED_YELLOW D8
#define BUZZER_PIN D2

// ── Constants ───────────────────────────────────────────────
#define DHT_TYPE   DHT22
#define BAUD_RATE  115200
#define SAMPLE_MS  2000      // Send reading every 2 seconds (prevent server overload)
#define WARMUP_SEC 30

// ── CD4051 channel map ─────────────────────────────────────
const uint8_t MUX_CH[5][3] = {
  {0,0,0},  // CH0 → MQ-135
  {1,0,0},  // CH1 → MQ-2
  {0,1,0},  // CH2 → MQ-7
  {1,1,0},  // CH3 → MQ-4
  {0,0,1}   // CH4 → MQ-3
};

// ── State ────────────────────────────────────────────────────
bool sending      = false;
bool wifiConnected= false;
unsigned long lastSample = 0;
unsigned long sampleCount = 0;
String lastPrediction = "";
String lastAlert = "";

DHT dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);
WiFiClient wifiClient;
HTTPClient httpClient;

// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);
  delay(100);

  pinMode(S0_PIN, OUTPUT);
  pinMode(S1_PIN, OUTPUT);
  pinMode(S2_PIN, OUTPUT);
  pinMode(FLAME_PIN, INPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  dht.begin();
  Wire.begin();
  lcd.init();
  lcd.backlight();

  Serial.println("\n\n========== SmartAir Guardian HTTP Client ==========");
  Serial.println("Connecting to WiFi...");
  
  lcdStatus("WiFi Connect", "...");
  connectWiFi();

  Serial.println("Warming up sensors...");
  lcdStatus("Warming up", String(WARMUP_SEC) + "s");
  
  for (int i = WARMUP_SEC; i > 0; i--) {
    digitalWrite(LED_GREEN, (i % 2 == 0));
    delay(1000);
  }
  digitalWrite(LED_GREEN, LOW);

  if (wifiConnected) {
    Serial.println("Ready to send predictions!");
    lcdStatus("READY", "WiFi OK");
    beep(2);
  } else {
    Serial.println("ERROR: WiFi failed!");
    lcdStatus("NO WiFi", "Check config");
    beep(5);
  }
}

// ─────────────────────────────────────────────────────────────
void loop() {
  // Check WiFi connection
  if (!WiFi.isConnected()) {
    wifiConnected = false;
    digitalWrite(LED_YELLOW, HIGH);  // Yellow = WiFi lost
    if (millis() % 5000 == 0) {
      Serial.println("[WARN] WiFi disconnected, attempting reconnect...");
      connectWiFi();
    }
  } else {
    wifiConnected = true;
    digitalWrite(LED_YELLOW, LOW);
  }

  // Handle Serial commands
  handleCommands();

  // Send prediction
  if (sending && wifiConnected && (millis() - lastSample >= SAMPLE_MS)) {
    lastSample = millis();
    collectAndPredict();
  }
}

// ─────────────────────────────────────────────────────────────
void collectAndPredict() {
  // Read sensors
  int mq135 = readMux(0);
  int mq2   = readMux(1);
  int mq7   = readMux(2);
  int mq4   = readMux(3);
  int mq3   = readMux(4);
  float temp = dht.readTemperature();
  float hum  = dht.readHumidity();
  int flame = (digitalRead(FLAME_PIN) == LOW) ? 1 : 0;

  if (isnan(temp) || isnan(hum)) {
    Serial.println("[ERR] DHT22 read failed");
    return;
  }

  // Normalize sensor values (map 0-1023 to 0.0-100.0 ppm equivalent)
  float mq135_ppm = (mq135 / 1023.0) * 100.0;
  float mq2_ppm   = (mq2   / 1023.0) * 100.0;
  float mq7_ppm   = (mq7   / 1023.0) * 100.0;
  float mq4_ppm   = (mq4   / 1023.0) * 100.0;
  float mq3_ppm   = (mq3   / 1023.0) * 100.0;

  sampleCount++;

  // Build JSON payload
  StaticJsonDocument<256> doc;
  doc["mq135"] = mq135_ppm;
  doc["mq3"]   = mq3_ppm;
  doc["mq7"]   = mq7_ppm;
  doc["mq4"]   = mq4_ppm;
  doc["temp"]  = temp;
  doc["hum"]   = hum;
  doc["flame"] = flame;

  String payload;
  serializeJson(doc, payload);

  Serial.print("[SEND #");
  Serial.print(sampleCount);
  Serial.print("] ");
  Serial.println(payload);

  // Send HTTP POST
  sendPredictionRequest(payload);

  // LED feedback
  digitalWrite(LED_GREEN, (sampleCount % 2 == 0));
}

// ─────────────────────────────────────────────────────────────
void sendPredictionRequest(String jsonPayload) {
  String url = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT) + API_ENDPOINT;
  
  httpClient.begin(wifiClient, url);
  httpClient.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = httpClient.POST(jsonPayload);
  
  if (httpResponseCode == 200) {
    String response = httpClient.getString();
    parseAndDisplayResponse(response);
    Serial.print("[OK] Response: ");
    Serial.println(response);
  } else {
    Serial.print("[ERR] HTTP response: ");
    Serial.println(httpResponseCode);
    lcdStatus("ERROR", "HTTP " + String(httpResponseCode));
  }
  
  httpClient.end();
}

// ─────────────────────────────────────────────────────────────
void parseAndDisplayResponse(String jsonResponse) {
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, jsonResponse);
  
  if (error) {
    Serial.print("[JSON ERR] ");
    Serial.println(error.f_str());
    return;
  }

  String gasClass = doc["prediction"]["gas_class"];
  float ppm = doc["prediction"]["ppm"];
  int severity = doc["prediction"]["severity"];
  
  String alert_type = doc["alert"].isNull() ? "NORMAL" : (const char*)doc["alert"]["type"];

  // Display on LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(gasClass);
  lcd.print(" ");
  lcd.print(ppm, 1);
  lcd.print("ppm");
  
  lcd.setCursor(0, 1);
  lcd.print("Alert: ");
  lcd.print(alert_type);

  // Audio alert for HIGH severity
  if (severity == 2) {
    beep(3);  // Three beeps for high alert
  }

  lastPrediction = gasClass;
  lastAlert = alert_type;
}

// ─────────────────────────────────────────────────────────────
void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.isConnected()) {
    wifiConnected = true;
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    lcdStatus("WiFi OK", WiFi.localIP().toString().c_str());
  } else {
    wifiConnected = false;
    Serial.println("\nFailed to connect to WiFi");
    lcdStatus("WiFi FAIL", "Check SSID/PWD");
  }
}

// ─────────────────────────────────────────────────────────────
int readMux(int channel) {
  digitalWrite(S0_PIN, MUX_CH[channel][0]);
  digitalWrite(S1_PIN, MUX_CH[channel][1]);
  digitalWrite(S2_PIN, MUX_CH[channel][2]);
  delay(10);
  return analogRead(A0);
}

// ─────────────────────────────────────────────────────────────
void handleCommands() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "START" && wifiConnected) {
    sending = true;
    Serial.println("[LIVE] Starting predictions...");
    lcdStatus("SENDING", "LIVE");
    beep(1);

  } else if (cmd == "STOP") {
    sending = false;
    Serial.println("[STOP] Predictions paused");
    lcdStatus("STOPPED", "Send START");
    beep(2);

  } else if (cmd == "STATUS") {
    Serial.print("WiFi: ");
    Serial.println(wifiConnected ? "CONNECTED" : "DISCONNECTED");
    Serial.print("Sending: ");
    Serial.println(sending ? "YES" : "NO");
    Serial.print("Samples sent: ");
    Serial.println(sampleCount);
    Serial.print("Last prediction: ");
    Serial.println(lastPrediction);

  } else if (cmd == "RECONNECT") {
    Serial.println("Reconnecting WiFi...");
    WiFi.disconnect();
    connectWiFi();

  } else if (cmd.length() > 0) {
    Serial.println("Commands: START, STOP, STATUS, RECONNECT");
  }
}

// ─────────────────────────────────────────────────────────────
void updateLCD(const char* line1, const char* line2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(line1);
  lcd.setCursor(0, 1); lcd.print(line2);
}

void lcdStatus(const char* line1, const char* line2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(line1);
  lcd.setCursor(0, 1); lcd.print(line2);
}

void lcdStatus(const char* line1, String line2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(line1);
  lcd.setCursor(0, 1); lcd.print(line2.c_str());
}

void beep(int count) {
  for (int i = 0; i < count; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(100);
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
  }
}
