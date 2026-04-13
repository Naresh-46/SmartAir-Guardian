// ============================================================
//  SmartAir — Button-Based Labeling System
//  Ported from ESP32 → ESP8266 (NodeMCU / Wemos D1 Mini)
//
//  KEY DIFFERENCES vs ESP32 version:
//   - ESP8266 has only ONE analog pin (A0), 10-bit (0–1023)
//   - MQ sensors share A0 via an analog multiplexer (CD4051 / CD74HC4051)
//     OR you read them one at a time using a MUX chip
//   - No GPIO 25/26/27/32–36 on ESP8266 — buttons moved to D5/D6/D7
//   - Flame sensor moved to D0 (digital only on ESP8266)
//   - INPUT_PULLUP works the same way (button pressed = LOW)
//
//  WIRING (NodeMCU pin labels):
//   Button 1 (Gas class) → D5  (GPIO14)
//   Button 2 (Severity)  → D6  (GPIO12)
//   Button 3 (Event)     → D7  (GPIO13)
//   Each button: one leg to GPIO pin, other leg to GND
//
//  MQ SENSORS via CD4051 8-channel analog MUX:
//   MUX S0 → D1 (GPIO5)
//   MUX S1 → D2 (GPIO4)
//   MUX S2 → D3 (GPIO0)  ← Note: GPIO0 is boot pin; keep HIGH at boot
//   MUX SIG (output) → A0
//   MUX channels:  Y0=MQ135, Y1=MQ3, Y2=MQ7, Y3=MQ4
//
//  Flame sensor (digital) → D0 (GPIO16)
//  DHT22 data pin         → D4 (GPIO2)
//
//  If you do NOT have a MUX chip:
//   Connect only MQ135 to A0 and comment out readMux() calls.
//   Read the other MQ sensors with a separate Arduino / MCP3208 ADC.
// ============================================================

#include <Arduino.h>
// Uncomment when you add DHT22:
// #include <DHT.h>

// ── PIN DEFINITIONS (ESP8266 NodeMCU) ───────────────────────
#define BTN_GAS       14   // D5 — cycles gas class
#define BTN_SEVERITY  12   // D6 — cycles severity
#define BTN_EVENT     13   // D7 — toggles event on/off

// Analog MUX select lines (CD4051 / CD74HC4051)
#define MUX_S0         5   // D1
#define MUX_S1         4   // D2
#define MUX_S2         0   // D3  (GPIO0 — safe after boot)

// MUX channel assignments
#define CH_MQ135       0   // S2=0, S1=0, S0=0
#define CH_MQ3         1   // S2=0, S1=0, S0=1
#define CH_MQ7         2   // S2=0, S1=1, S0=0
#define CH_MQ4         3   // S2=0, S1=1, S0=1

#define PIN_FLAME      16  // D0 — digital: LOW = flame detected
#define PIN_DHT        2   // D4 — DHT22 data

// ESP8266 single ADC pin
#define ANALOG_PIN     A0

// ── LABEL STATE ─────────────────────────────────────────────
int  gasClass  = 0;   // 0=clean 1=smoke/CO 2=alcohol 3=NH3 4=fire 5=mixed
int  severity  = 0;   // 0=none  1=low  2=medium  3=high
int  eventID   = 0;   // increments each time event starts
bool inEvent   = false;

// Collection start time — used to mark warmup rows
unsigned long startMillis = 0;
const unsigned long WARMUP_MS = 24UL * 60 * 60 * 1000; // 24 hours

// Debounce
unsigned long lastBtn1 = 0;
unsigned long lastBtn2 = 0;
unsigned long lastBtn3 = 0;
const unsigned long DEBOUNCE_MS = 300;

// ── LABEL NAME TABLES ────────────────────────────────────────
const char* GAS_NAMES[] = {
  "0-CLEAN",
  "1-SMOKE_CO",
  "2-ALCOHOL_VOC",
  "3-NH3_AMMONIA",
  "4-FIRE_FLAME",
  "5-MIXED_LPG"
};

const char* SEV_NAMES[] = {
  "0-NONE",
  "1-LOW",
  "2-MEDIUM",
  "3-HIGH"
};

// ── DHT22 (uncomment when library is added) ──────────────────
// #define DHT_TYPE DHT22
// DHT dht(PIN_DHT, DHT_TYPE);

// ── ANALOG MUX READ ─────────────────────────────────────────
// Select a CD4051 channel and read A0
// ESP8266 ADC is 10-bit: returns 0–1023
int readMux(int channel) {
  digitalWrite(MUX_S0, (channel >> 0) & 1);
  digitalWrite(MUX_S1, (channel >> 1) & 1);
  digitalWrite(MUX_S2, (channel >> 2) & 1);
  delayMicroseconds(200); // settle time
  return analogRead(ANALOG_PIN);
}

// ── SETUP ────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(200);

  // Button pins — pressed = LOW
  pinMode(BTN_GAS,      INPUT_PULLUP);
  pinMode(BTN_SEVERITY, INPUT_PULLUP);
  pinMode(BTN_EVENT,    INPUT_PULLUP);

  // MUX select lines
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  digitalWrite(MUX_S0, LOW);
  digitalWrite(MUX_S1, LOW);
  digitalWrite(MUX_S2, LOW);

  // Flame sensor (digital input, no pull-up — sensor has its own)
  pinMode(PIN_FLAME, INPUT);

  // DHT22
  // dht.begin();

  startMillis = millis();

  // ── YOUR EXISTING SETUP CODE HERE ──
  // e.g. SD.begin(), WiFi.begin() etc.

  // CSV header
  Serial.println("timestamp,MQ135,MQ3,MQ7,MQ4,DHT_temp,DHT_hum,flame,"
                 "gas_class,severity,event_id,is_event,is_warmup");

  Serial.println("\n[SmartAir ESP8266] Labeling system ready.");
  printCurrentState();
}

// ── MAIN LOOP ────────────────────────────────────────────────
void loop() {
  checkButtons();
  logRow();
  delay(1000); // 1 second sampling rate
}

// ── BUTTON HANDLER ───────────────────────────────────────────
void checkButtons() {
  unsigned long now = millis();

  // Button 1 — cycle gas class
  if (digitalRead(BTN_GAS) == LOW && (now - lastBtn1) > DEBOUNCE_MS) {
    lastBtn1 = now;
    gasClass = (gasClass + 1) % 6;
    Serial.print("[BTN1] Gas class → ");
    Serial.println(GAS_NAMES[gasClass]);
    printCurrentState();
  }

  // Button 2 — cycle severity
  if (digitalRead(BTN_SEVERITY) == LOW && (now - lastBtn2) > DEBOUNCE_MS) {
    lastBtn2 = now;
    severity = (severity + 1) % 4;
    Serial.print("[BTN2] Severity  → ");
    Serial.println(SEV_NAMES[severity]);
    printCurrentState();
  }

  // Button 3 — toggle event on/off
  if (digitalRead(BTN_EVENT) == LOW && (now - lastBtn3) > DEBOUNCE_MS) {
    lastBtn3 = now;
    inEvent = !inEvent;
    if (inEvent) {
      eventID++;
      Serial.print("[BTN3] EVENT START → event_id=");
      Serial.println(eventID);
    } else {
      Serial.print("[BTN3] EVENT END   → event_id=");
      Serial.println(eventID);
      Serial.println("       (ventilate 90 sec before next event)");
    }
    printCurrentState();
  }
}

// ── DATA LOGGING ─────────────────────────────────────────────
void logRow() {
  // Read all 4 MQ sensors via analog MUX
  // ESP8266 ADC returns 0–1023 (10-bit, vs ESP32's 0–4095 12-bit)
  int mq135_raw = readMux(CH_MQ135);
  int mq3_raw   = readMux(CH_MQ3);
  int mq7_raw   = readMux(CH_MQ7);
  int mq4_raw   = readMux(CH_MQ4);

  // Flame sensor — digital LOW = flame detected
  int flame_val = (digitalRead(PIN_FLAME) == LOW) ? 1 : 0;

  // DHT22 — uncomment when library added
  float temp = 0.0; // dht.readTemperature();
  float hum  = 0.0; // dht.readHumidity();

  // Warmup flag
  int is_warmup = (millis() - startMillis < WARMUP_MS) ? 1 : 0;

  // Timestamp as millis (replace with RTC if available)
  unsigned long ts = millis();

  // Print CSV row to Serial
  Serial.print(ts);              Serial.print(",");
  Serial.print(mq135_raw);      Serial.print(",");
  Serial.print(mq3_raw);        Serial.print(",");
  Serial.print(mq7_raw);        Serial.print(",");
  Serial.print(mq4_raw);        Serial.print(",");
  Serial.print(temp, 2);        Serial.print(",");
  Serial.print(hum, 2);         Serial.print(",");
  Serial.print(flame_val);      Serial.print(",");
  Serial.print(gasClass);       Serial.print(",");
  Serial.print(severity);       Serial.print(",");
  Serial.print(eventID);        Serial.print(",");
  Serial.print(inEvent ? 1 : 0); Serial.print(",");
  Serial.println(is_warmup);

  // ── TO WRITE TO SD CARD ──────────────────────────────────
  // File f = SD.open("data.csv", FILE_APPEND);
  // if (f) {
  //   f.print(ts);              f.print(",");
  //   f.print(mq135_raw);      f.print(",");
  //   f.print(mq3_raw);        f.print(",");
  //   f.print(mq7_raw);        f.print(",");
  //   f.print(mq4_raw);        f.print(",");
  //   f.print(temp, 2);        f.print(",");
  //   f.print(hum, 2);         f.print(",");
  //   f.print(flame_val);      f.print(",");
  //   f.print(gasClass);       f.print(",");
  //   f.print(severity);       f.print(",");
  //   f.print(eventID);        f.print(",");
  //   f.print(inEvent ? 1 : 0); f.print(",");
  //   f.println(is_warmup);
  //   f.close();
  // }
}

// ── HELPER: print current labeling state ────────────────────
void printCurrentState() {
  Serial.println("─────────────────────────────");
  Serial.print("  Gas:      "); Serial.println(GAS_NAMES[gasClass]);
  Serial.print("  Severity: "); Serial.println(SEV_NAMES[severity]);
  Serial.print("  Event:    ");
  if (inEvent) {
    Serial.print("ACTIVE (id=");
    Serial.print(eventID);
    Serial.println(")");
  } else {
    Serial.println("INACTIVE");
  }
  Serial.println("─────────────────────────────");
}

// ============================================================
//  ESP8266 PIN MAPPING REFERENCE
//
//  NodeMCU Label | GPIO | This sketch
//  D0            | 16   | Flame sensor (digital)
//  D1            |  5   | MUX S0
//  D2            |  4   | MUX S1
//  D3            |  0   | MUX S2  (boot pin — keep HIGH at boot)
//  D4            |  2   | DHT22   (boot pin — has on-board LED)
//  D5            | 14   | Button 1 (Gas class)
//  D6            | 12   | Button 2 (Severity)
//  D7            | 13   | Button 3 (Event toggle)
//  A0            |  -   | MUX SIG (analog in, 0–1023, 10-bit)
//
//  IMPORTANT ESP8266 ADC NOTE:
//  ESP8266 has only ONE analog input (A0).
//  Max input voltage on A0 = 1.0V (NodeMCU has onboard divider → 3.3V)
//  MQ sensor output is 0–5V typically — use a voltage divider if VCC=5V.
//  If powering MQ sensors at 3.3V, NodeMCU A0 is safe directly.
//
//  GPIO0 (D3) BOOT NOTE:
//  GPIO0 must be HIGH at power-on for normal boot.
//  Buttons on D3 are not used here. MUX S2 defaults LOW (fine for boot).
//  Do NOT connect a button to D3 without a pull-up resistor.
//
//  WITHOUT A MUX CHIP:
//  Connect only MQ135 to A0. For other sensors, either:
//    a) Add an MCP3208 SPI ADC chip (8 channels, 12-bit, SPI)
//    b) Use a CD4051 analog MUX (cheapest option, ~₹20)
//    c) Upgrade to ESP32 which has multiple ADC pins built-in
//
// ============================================================
//  QUICK USAGE GUIDE (same as ESP32 version)
//
//  BEFORE each gas exposure:
//    1. Press BTN1 (D5) to set gas class
//    2. Press BTN2 (D6) to set severity
//    3. Introduce gas source, then press BTN3 (D7) → EVENT START
//
//  AFTER removing gas source:
//    4. Press BTN3 → EVENT END
//    5. Wait 90 seconds before next exposure
//    6. Reset BTN1 → class 0, BTN2 → severity 0
//
//  GAS CLASS:  0=Clean  1=Smoke/CO  2=Alcohol  3=NH3  4=Fire  5=Mixed
//  SEVERITY:   0=None   1=Low       2=Medium   3=High
// ============================================================
