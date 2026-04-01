// ============================================================
//  SmartAir Guardian — Dataset Collection Firmware
//  ESP8266 NodeMCU + CD4051 + MQ135/2/7/4/3 + DHT22
//  Sends CSV rows over Serial for Python to capture
// ============================================================

#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

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
#define SAMPLE_MS  500        // one row every 500ms
#define WARMUP_SEC 30         // sensor warm-up before collection

// ── CD4051 channel map: channel → {S0, S1, S2} ─────────────
const uint8_t MUX_CH[5][3] = {
  {0,0,0},  // CH0 → MQ-135
  {1,0,0},  // CH1 → MQ-2
  {0,1,0},  // CH2 → MQ-7
  {1,1,0},  // CH3 → MQ-4
  {0,0,1}   // CH4 → MQ-3
};

// ── Gas class labels (set via Serial command) ────────────────
// 0=Normal  1=LPG  2=Smoke  3=CO  4=Methane
int currentLabel = 0;
const char* LABEL_NAMES[] = {
  "NORMAL", "LPG", "SMOKE", "CO", "METHANE"
};

// ── State ────────────────────────────────────────────────────
bool collecting   = false;
bool headerPrinted = false;
unsigned long lastSample = 0;
unsigned long sampleCount = 0;
unsigned long sessionCount[5] = {0,0,0,0,0};

DHT dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);  // change to 0x3F if needed

// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(BAUD_RATE);
  delay(100);

  // Pin setup
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

  // Print startup banner to Serial (Python ignores non-CSV lines)
  Serial.println("# =========================================");
  Serial.println("# SmartAir Guardian — Dataset Collector");
  Serial.println("# Commands:");
  Serial.println("#   START   — begin collecting");
  Serial.println("#   STOP    — pause collecting");
  Serial.println("#   LABEL:0 — set class to Normal");
  Serial.println("#   LABEL:1 — set class to LPG");
  Serial.println("#   LABEL:2 — set class to Smoke");
  Serial.println("#   LABEL:3 — set class to CO");
  Serial.println("#   LABEL:4 — set class to Methane");
  Serial.println("#   STATUS  — print current state");
  Serial.println("#   COUNT   — print sample counts");
  Serial.println("# =========================================");

  // Warm-up phase
  lcdStatus("Warming up...", "Please wait");
  Serial.println("# Warming up sensors for " + String(WARMUP_SEC) + " seconds...");

  for (int i = WARMUP_SEC; i > 0; i--) {
    digitalWrite(LED_GREEN, (i % 2 == 0));
    lcd.setCursor(0, 1);
    lcd.print("Ready in: ");
    lcd.print(i);
    lcd.print("s  ");
    delay(1000);
  }
  digitalWrite(LED_GREEN, LOW);

  Serial.println("# Sensors ready. Send START to begin.");
  lcdStatus("READY", "Send START");
}

// ─────────────────────────────────────────────────────────────
void loop() {
  // Handle Serial commands from Python
  handleCommands();

  // Collect sample
  if (collecting && (millis() - lastSample >= SAMPLE_MS)) {
    lastSample = millis();
    collectAndSend();
  }
}

// ─────────────────────────────────────────────────────────────
void collectAndSend() {
  // Read all MQ sensors via CD4051
  int mq135 = readMux(0);
  int mq2   = readMux(1);
  int mq7   = readMux(2);
  int mq4   = readMux(3);
  int mq3   = readMux(4);

  // Read DHT22
  float temp = dht.readTemperature();
  float hum  = dht.readHumidity();

  // Read flame sensor (1=flame detected, 0=no flame)
  int flame = (digitalRead(FLAME_PIN) == LOW) ? 1 : 0;

  // Skip if DHT22 failed
  if (isnan(temp) || isnan(hum)) {
    Serial.println("# DHT22 read failed — skipping row");
    return;
  }

  // Print CSV header once
  if (!headerPrinted) {
    Serial.println("mq135,mq2,mq7,mq4,mq3,temperature,humidity,flame,label,label_name,timestamp_ms");
    headerPrinted = true;
  }

  // Print data row — Python reads this
  sampleCount++;
  sessionCount[currentLabel]++;

  Serial.print(mq135);       Serial.print(",");
  Serial.print(mq2);         Serial.print(",");
  Serial.print(mq7);         Serial.print(",");
  Serial.print(mq4);         Serial.print(",");
  Serial.print(mq3);         Serial.print(",");
  Serial.print(temp, 1);     Serial.print(",");
  Serial.print(hum, 1);      Serial.print(",");
  Serial.print(flame);       Serial.print(",");
  Serial.print(currentLabel);Serial.print(",");
  Serial.print(LABEL_NAMES[currentLabel]); Serial.print(",");
  Serial.println(millis());

  // Update LCD every row
  updateLCD(mq135, mq2, temp, hum);

  // LED feedback
  digitalWrite(LED_GREEN, (sampleCount % 2 == 0));
}

// ─────────────────────────────────────────────────────────────
int readMux(int channel) {
  digitalWrite(S0_PIN, MUX_CH[channel][0]);
  digitalWrite(S1_PIN, MUX_CH[channel][1]);
  digitalWrite(S2_PIN, MUX_CH[channel][2]);
  delay(10);  // CD4051 settling time
  return analogRead(A0);
}

// ─────────────────────────────────────────────────────────────
void handleCommands() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "START") {
    collecting = true;
    Serial.println("# COLLECTING — Label: " + String(currentLabel) +
                   " (" + String(LABEL_NAMES[currentLabel]) + ")");
    lcdStatus("COLLECTING", LABEL_NAMES[currentLabel]);
    digitalWrite(LED_GREEN, HIGH);
    beep(1);

  } else if (cmd == "STOP") {
    collecting = false;
    Serial.println("# STOPPED — Total rows this session: " + String(sampleCount));
    lcdStatus("STOPPED", "Send START");
    digitalWrite(LED_GREEN, LOW);
    beep(2);

  } else if (cmd.startsWith("LABEL:")) {
    int lbl = cmd.substring(6).toInt();
    if (lbl >= 0 && lbl <= 4) {
      currentLabel = lbl;
      Serial.println("# Label changed to: " + String(lbl) +
                     " (" + String(LABEL_NAMES[lbl]) + ")");
      lcdStatus("Label set:", LABEL_NAMES[lbl]);
      beep(1);
    } else {
      Serial.println("# Invalid label. Use 0–4.");
    }

  } else if (cmd == "STATUS") {
    Serial.println("# Status: " + String(collecting ? "COLLECTING" : "STOPPED"));
    Serial.println("# Current label: " + String(currentLabel) +
                   " (" + String(LABEL_NAMES[currentLabel]) + ")");
    Serial.println("# Total samples: " + String(sampleCount));

  } else if (cmd == "COUNT") {
    Serial.println("# Sample counts per class:");
    for (int i = 0; i < 5; i++) {
      Serial.println("#   " + String(LABEL_NAMES[i]) + ": " +
                     String(sessionCount[i]) + " rows");
    }

  } else if (cmd.length() > 0) {
    Serial.println("# Unknown command: " + cmd);
  }
}

// ─────────────────────────────────────────────────────────────
void updateLCD(int mq135, int mq2, float temp, float hum) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(LABEL_NAMES[currentLabel]);
  lcd.print(" #");
  lcd.print(sampleCount);

  lcd.setCursor(0, 1);
  lcd.print("T:");
  lcd.print(temp, 1);
  lcd.print(" H:");
  lcd.print(hum, 0);
  lcd.print("%");
}

void lcdStatus(const char* line1, const char* line2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(line1);
  lcd.setCursor(0, 1); lcd.print(line2);
}

void beep(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(80);
    digitalWrite(BUZZER_PIN, LOW);  delay(80);
  }
}
