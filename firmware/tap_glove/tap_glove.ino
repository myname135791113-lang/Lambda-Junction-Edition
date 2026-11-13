// tap_glove.ino: capture fingertip taps and stream them to glove_twin over USB serial.
//
// Board: Teensy 4.x recommended (fast 12-bit ADC). ESP32 also works at a lower FS.
// An Arduino Uno is too slow for 3 channels at 16 kHz; set FS to 8000 and drop the mic there.
//
// Wiring (3.3 V logic):
//   PIEZO (fingertip, under a fingernail or a hard thimble)
//     - one leg to a mid-rail reference: 3V3 -- 100k -- REF -- 100k -- GND, 10 uF REF to GND
//     - other leg -- 1k -- A0;  1M from A0 to REF (bleed); the 1k plus the MCU clamp diodes protect the pin
//   FSR 402 (same fingertip, under the piezo)
//     - 3V3 -- FSR -- A1 -- 10k -- GND
//   MIC (optional but recommended): MAX4466 electret module on the back of the hand
//     - OUT -- A2.  Use a FIXED-gain module: AGC mics (MAX9814) distort the decay measurement.
//
// Each tap is sent as one JSON line:
//   {"fs":16000,"pre":64,"adc_bits":12,"piezo":[...],"fsr":[...],"mic":[...]}

#define USE_MIC 1

const int PIEZO_PIN = A0;
const int FSR_PIN = A1;
const int MIC_PIN = A2;

const uint32_t FS = 16000;              // samples per second per channel
const uint32_t PERIOD_US = 1000000UL / FS;
const int N = 1024;                     // samples per tap window (64 ms at 16 kHz)
const int PRE = 64;                     // samples kept from before the trigger
const int ADC_BITS = 12;
const int TRIGGER_COUNTS = 120;         // piezo deviation from baseline that counts as a tap
const uint32_t REFRACTORY_MS = 150;     // ignore the tail of the same tap

uint16_t piezoBuf[N], fsrBuf[N], micBuf[N];
uint16_t ringP[PRE], ringF[PRE], ringM[PRE];
int ringIdx = 0;
float baseline = 0;

static inline void readAll(uint16_t &p, uint16_t &f, uint16_t &m) {
  p = analogRead(PIEZO_PIN);
  f = analogRead(FSR_PIN);
#if USE_MIC
  m = analogRead(MIC_PIN);
#else
  m = 0;
#endif
}

static void fillRing() {
  uint32_t next = micros();
  for (int i = 0; i < PRE; i++) {
    while ((int32_t)(micros() - next) < 0) {}
    next += PERIOD_US;
    readAll(ringP[i], ringF[i], ringM[i]);
  }
  ringIdx = 0;
}

static void printArray(const char *key, const uint16_t *buf) {
  Serial.print(",\""); Serial.print(key); Serial.print("\":[");
  for (int i = 0; i < N; i++) {
    if (i) Serial.print(',');
    Serial.print(buf[i]);
  }
  Serial.print(']');
}

static void sendPacket() {
  Serial.print("{\"fs\":"); Serial.print(FS);
  Serial.print(",\"pre\":"); Serial.print(PRE);
  Serial.print(",\"adc_bits\":"); Serial.print(ADC_BITS);
  printArray("piezo", piezoBuf);
  printArray("fsr", fsrBuf);
#if USE_MIC
  printArray("mic", micBuf);
#endif
  Serial.println('}');
}

static void capture(uint16_t p0, uint16_t f0, uint16_t m0) {
  for (int i = 0; i < PRE; i++) {        // pre-trigger history, oldest first
    int j = (ringIdx + i) % PRE;
    piezoBuf[i] = ringP[j]; fsrBuf[i] = ringF[j]; micBuf[i] = ringM[j];
  }
  piezoBuf[PRE] = p0; fsrBuf[PRE] = f0; micBuf[PRE] = m0;
  uint32_t next = micros() + PERIOD_US;
  for (int i = PRE + 1; i < N; i++) {
    while ((int32_t)(micros() - next) < 0) {}
    next += PERIOD_US;
    readAll(piezoBuf[i], fsrBuf[i], micBuf[i]);
  }
  sendPacket();
  delay(REFRACTORY_MS);
  fillRing();
}

void setup() {
  Serial.begin(921600);
  analogReadResolution(ADC_BITS);
#if defined(TEENSYDUINO)
  analogReadAveraging(1);                // speed over smoothing; we need the transient
#endif
  delay(500);
  uint32_t sum = 0;
  for (int i = 0; i < 2000; i++) sum += analogRead(PIEZO_PIN);
  baseline = sum / 2000.0f;
  fillRing();
  Serial.print("tap_glove ready, baseline ");
  Serial.println(baseline);
}

void loop() {
  static uint32_t next = micros();
  while ((int32_t)(micros() - next) < 0) {}
  next += PERIOD_US;

  uint16_t p, f, m;
  readAll(p, f, m);
  if (abs((int)p - (int)baseline) > TRIGGER_COUNTS) {
    capture(p, f, m);
    next = micros();
    return;
  }
  baseline += (p - baseline) * 0.001f;   // track slow drift while idle
  ringP[ringIdx] = p; ringF[ringIdx] = f; ringM[ringIdx] = m;
  ringIdx = (ringIdx + 1) % PRE;
}
