# ✅ ESP to Server to ML Pipeline

## Setup Instructions

### **1. Get Your Laptop IP Address**

**Windows PowerShell:**
```powershell
ipconfig
```
Look for "IPv4 Address" under your WiFi connection (e.g., `192.168.x.x`)

**Example output:**
```
Ethernet adapter WiFi:
   IPv4 Address . . . . . . . . . . . : 192.168.1.105
```

---

### **2. Configure ESP Firmware**

Open the file: `data_collection/firmware/smartair_http_client/smartair_http_client.ino`

Edit **Line 12-14** with your WiFi details:

```cpp
const char* WIFI_SSID     = "YOUR_SSID";           // Your WiFi network name
const char* WIFI_PASSWORD = "YOUR_PASSWORD";      // Your WiFi password
const char* SERVER_IP     = "192.168.1.105";      // Your laptop IPv4 address
```

**Example for home WiFi:**
```cpp
const char* WIFI_SSID     = "MyHomeWiFi";
const char* WIFI_PASSWORD = "MyPassword123";
const char* SERVER_IP     = "192.168.1.105";
```

---

### **3. Upload Firmware to ESP**

**In Arduino IDE:**
1. Copy entire content from `smartair_http_client.ino`
2. Create new sketch
3. Paste code
4. Install libraries if prompted:
   - `ESP8266WiFi`
   - `ESP8266HTTPClient`
   - `ArduinoJson` (by Benoit Blanchon)
   - `DHT` (by Adafruit)
   - `LiquidCrystal_I2C` (by Frank de Brabander)

5. Select Board: **NodeMCU 1.0 (ESP-12E Module)**
6. Select Port: Your COM port
7. Click **Upload** (Sketch → Upload)

---

### **4. Ensure Flask Server is Running**

In PowerShell (SmartAir-Guardian folder):
```bash
python -m server.app
```

Expected output:
```
[MODEL] ✓  Loaded from model/outputs/smartair_model.keras
[SmartAir] Server starting → http://localhost:5000
 * Running on http://127.0.0.1:5000
```

---

### **5. Connect ESP & Start Sending Data**

**In Arduino Serial Monitor (115200 baud):**

```
===== SmartAir Guardian HTTP Client =====
Connecting to WiFi...
WiFi connected!
IP address: 192.168.1.106
Ready to send predictions!

> START
[LIVE] Starting predictions...

[SEND #1] {"mq135":25.3, "mq3":15.2, "mq7":8.1, "mq4":42.0, "temp":23.5, "hum":45.2, "flame":0}
[OK] Response: {"timestamp":"2026-04-13T20:45:33","prediction":{"gas_class":"Normal","ppm":28.5,"severity":0},"alert":null}

[SEND #2] {"mq135":26.1, ...}
[OK] Response: ...
```

---

## **Data Flow** 🔄

```
ESP8266 Sensors
    ↓ (WiFi)
Flask Server (/api/predict)
    ↓
Trained ML Model
    ↓
Predictions + Alerts
    ↓
ESP8266 LCD Display & Alerts
```

---

## **Commands via Serial Monitor**

| Command | Action |
|---------|--------|
| `START` | Begin sending sensor readings to server |
| `STOP` | Pause predictions |
| `STATUS` | Show WiFi/sending status & stats |
| `RECONNECT` | Force WiFi reconnection |

---

## **LED Status**

| LED | State | Meaning |
|-----|-------|---------|
| 🟢 Green | Blinking | Sending data |
| 🟢 Green | Off | Stopped |
| 🟡 Yellow | On | WiFi disconnected |
| 🔴 Red | N/A | (Reserved for future use) |

---

## **Troubleshooting**

**"WiFi FAIL"**
- Check SSID/password spelling
- Ensure laptop WiFi is on (not cellular)
- Try `RECONNECT` command

**"HTTP 500"**
- Flask server crashed → restart with `python -m server.app`
- Check server terminal for errors

**"No Response"**
- Verify server is running on port 5000
- Check firewall (allow port 5000)
- Ping test: `ping 192.168.1.105`

**Serial garbage output**
- Check baud rate: **must be 115200**
- Try different USB cable

---

## **Notes**

- Predictions sent every **2 seconds** (configurable at `#define SAMPLE_MS`)
- Server stores predictions in history (viewable at http://localhost:5000)
- Model runs locally on your laptop (not in cloud)
- Alert thresholds in `server/utils/alerts.py`
