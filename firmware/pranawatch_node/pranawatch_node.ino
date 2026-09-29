/*
 * PranaWatch field node - ESP32 + PMS5003 (PM1/PM2.5/PM10) + BME280 (T/RH/P)
 *
 * Wiring:
 *   PMS5003 TX  -> ESP32 GPIO16 (UART2 RX)   PMS5003 VCC -> 5V, GND -> GND
 *   PMS5003 SET -> ESP32 GPIO4 (HIGH = run, LOW = sleep, saves fan life)
 *   BME280 SDA  -> GPIO21, SCL -> GPIO22, VCC -> 3V3
 *
 * Every 60 s: wake PMS5003, wait 30 s for fan to stabilise, average 10 frames,
 * read BME280, POST an OGC SensorThings-style JSON to the ingest endpoint.
 * Raw values are sent; humidity calibration happens in the cloud (core/calibration.py)
 * so the model can be improved without re-flashing nodes.
 *
 * Libraries: Adafruit BME280, ArduinoJson (v7)
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_BME280.h>
#include <ArduinoJson.h>

#define NODE_ID      "PW-01"
#define WIFI_SSID    "YOUR_WIFI"
#define WIFI_PASS    "YOUR_PASSWORD"
// e.g. Firebase Realtime DB REST: https://<project>-default-rtdb.asia-southeast1.firebasedatabase.app/obs.json
#define INGEST_URL   "https://YOUR_INGEST_ENDPOINT/obs.json"

#define PMS_RX 16
#define PMS_TX 17
#define PMS_SET 4

HardwareSerial pms(2);
Adafruit_BME280 bme;

struct PmFrame { uint16_t pm1, pm25, pm10; };

bool readPmsFrame(PmFrame &f, uint32_t timeoutMs = 2000) {
  uint8_t buf[32];
  uint32_t t0 = millis();
  while (millis() - t0 < timeoutMs) {
    if (pms.available() >= 32 && pms.peek() == 0x42) {
      pms.readBytes(buf, 32);
      if (buf[1] != 0x4D) continue;
      uint16_t sum = 0;
      for (int i = 0; i < 30; i++) sum += buf[i];
      if (sum != ((buf[30] << 8) | buf[31])) continue;   // checksum
      f.pm1  = (buf[10] << 8) | buf[11];                 // atmospheric environment values
      f.pm25 = (buf[12] << 8) | buf[13];
      f.pm10 = (buf[14] << 8) | buf[15];
      return true;
    } else if (pms.available() && pms.peek() != 0x42) {
      pms.read();                                        // resync
    }
    delay(5);
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  pms.begin(9600, SERIAL_8N1, PMS_RX, PMS_TX);
  pinMode(PMS_SET, OUTPUT);
  digitalWrite(PMS_SET, LOW);
  if (!bme.begin(0x76)) Serial.println("BME280 not found!");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi OK");
  configTime(19800, 0, "pool.ntp.org");                 // IST
}

void loop() {
  digitalWrite(PMS_SET, HIGH);                          // wake sensor
  delay(30000);                                         // fan spin-up
  uint32_t s1 = 0, s25 = 0, s10 = 0; int n = 0;
  PmFrame f;
  while (n < 10) { if (readPmsFrame(f)) { s1 += f.pm1; s25 += f.pm25; s10 += f.pm10; n++; } }
  digitalWrite(PMS_SET, LOW);                           // sleep sensor

  JsonDocument doc;
  doc["node_id"] = NODE_ID;
  doc["time"] = (long)time(nullptr);
  doc["pm1_raw"] = s1 / (float)n;
  doc["pm25_raw"] = s25 / (float)n;
  doc["pm10_raw"] = s10 / (float)n;
  doc["temp"] = bme.readTemperature();
  doc["rh"] = bme.readHumidity();
  doc["pressure_hpa"] = bme.readPressure() / 100.0F;
  String body; serializeJson(doc, body);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(INGEST_URL);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST(body);
    Serial.printf("POST %d  %s\n", code, body.c_str());
    http.end();
  }
  delay(30000);                                         // ~60 s cycle
}
