#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// -------------------------------
// Configuration
// -------------------------------
#define BUTTON_PIN          4
#define CALIB_SWITCH_PIN    15
#define DEBOUNCE_MS         50
// #define HOLD_CALIB_MS       5000   // hold duration to send CALIBRATE
#define SERIAL_BAUD         115200
#define MIN_CALIB_FRAMES    10     // must match scan.py
#define SCAN_TIMEOUT_MS     180000 // scan + reconstruction + roughness can take a while
#define CAPTURE_TIMEOUT_MS  3000   // if no FRAME response within 3s, show error (Pi uses 1s camera timeout)

LiquidCrystal_I2C lcd(0x27, 16, 2);

// -------------------------------
// State
// -------------------------------
bool          lastButtonState  = HIGH;
bool          buttonPressed    = false;
// bool          holdTriggered    = false;
unsigned long lastDebounceTime = 0;
// unsigned long pressStartTime   = 0;
// unsigned long lastHoldUpdateMs = 0;

bool          lastSwitchState    = HIGH;  // HIGH = open = OFF (INPUT_PULLUP)
bool          switchState        = HIGH;
unsigned long lastSwitchDebounce = 0;

bool calibModeActive = false;
int  calibFrameCount = 0;

// Response tracking — reset to idle if Pi goes quiet
bool          waitingForScan    = false;
unsigned long scanSentTime      = 0;
bool          waitingForCapture = false;
unsigned long captureSentTime   = 0;
bool          resultDisplayed   = false;

// -------------------------------
// LCD helpers
// -------------------------------
void lcdPrint(const char* line0, const char* line1) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line0);
  lcd.setCursor(0, 1);
  lcd.print(line1);
}

void showIdle() {
  lcdPrint("  3D Scanner    ", "Ready to Scan   ");
}

String valueAfterKey(const String& msg, const char* key) {
  int start = msg.indexOf(key);
  if (start < 0) return "";
  start += strlen(key);
  int end = msg.indexOf(' ', start);
  if (end < 0) end = msg.length();
  return msg.substring(start, end);
}

void showResult(const String& msg) {
  String sa    = valueAfterKey(msg, "SA=");
  String sq    = valueAfterKey(msg, "SQ=");
  String svr   = valueAfterKey(msg, "SVR=");
  String noise = valueAfterKey(msg, "NF=");
  if (noise.length() == 0) {
    noise = valueAfterKey(msg, "NOISE=");
  }

  char line0[17];
  char line1[17];
  snprintf(line0, sizeof(line0), "Sa%s Sq%s", sa.c_str(), sq.c_str());
  if (noise.length() > 0 && noise != "0.0") {
    snprintf(line1, sizeof(line1), "Svr%s NF%s", svr.c_str(), noise.c_str());
  } else {
    snprintf(line1, sizeof(line1), "Svr%s um", svr.c_str());
  }
  lcdPrint(line0, line1);
  resultDisplayed = true;
}

// -------------------------------
// Setup
// -------------------------------
void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(CALIB_SWITCH_PIN, INPUT_PULLUP);
  // Read initial switch state so boot position doesn't send a spurious CALIBRATE
  switchState     = digitalRead(CALIB_SWITCH_PIN);
  lastSwitchState = switchState;
  Serial.begin(SERIAL_BAUD);
  lcd.init();
  lcd.backlight();
  showIdle();
}

// -------------------------------
// Loop
// -------------------------------
void loop() {
  unsigned long now = millis();

  // ----- Calib switch (pin 15) -----
  bool switchReading = digitalRead(CALIB_SWITCH_PIN);
  if (switchReading != lastSwitchState) {
    lastSwitchDebounce = now;
  }
  if (now - lastSwitchDebounce > DEBOUNCE_MS && switchReading != switchState) {
    switchState = switchReading;
    if (switchState == LOW) {   // switch turned ON → enter calibration mode
      Serial.println("CALIB_ON");
      lcdPrint("Calibration mode", "Please wait...  ");
    } else {                    // switch turned OFF → exit calibration mode
      Serial.println("CALIB_OFF");
      lcdPrint("Finalizing...   ", "Please wait...  ");
    }
  }
  lastSwitchState = switchReading;

  // ----- Button debounce (pin 4) -----
  bool reading = digitalRead(BUTTON_PIN);
  if (reading != lastButtonState) {
    lastDebounceTime = now;
  }

  if (now - lastDebounceTime > DEBOUNCE_MS) {

    if (reading == LOW && !buttonPressed) {
      // Press start
      buttonPressed    = true;
      // holdTriggered    = false;
      // pressStartTime   = now;
      // lastHoldUpdateMs = now;

      if (switchState == LOW || calibModeActive) {
        lcdPrint("Capture frame   ", "Release=Capture ");
      } else {
        lcdPrint("Button held...  ", "Release=Scan    ");
      }

    } else if (reading == HIGH && buttonPressed) {
      // Release
      buttonPressed = false;

      // if (!holdTriggered) {
        if (switchState == LOW || calibModeActive) {
          Serial.println("CALIB_TRIGGER");
          lcdPrint("Capturing...    ", "                ");
          waitingForCapture = true;
          captureSentTime   = now;
        } else {
          Serial.println("TRIGGER");
          lcdPrint("Scanning...     ", "Hold still...   ");
          waitingForScan  = true;
          scanSentTime    = now;
          resultDisplayed = false;
        }
      // }
    }
  }

  lastButtonState = reading;

  // ----- Live hold countdown (updates line 1 only, no full clear) -----
  // if (buttonPressed && !holdTriggered) {
  //   unsigned long heldMs = now - pressStartTime;
  //
  //   if (heldMs >= HOLD_CALIB_MS) {
  //     Serial.println("CALIBRATE");
  //     holdTriggered = true;
  //     if (calibModeActive) {
  //       lcdPrint("Discarding...   ", "Please wait...  ");
  //     } else {
  //       lcdPrint("Calibration mode", "Please wait...  ");
  //     }
  //   } else if (now - lastHoldUpdateMs >= 200) {
  //     lastHoldUpdateMs = now;
  //     unsigned int remaining = (unsigned int)((HOLD_CALIB_MS - heldMs) / 1000) + 1;
  //     char buf[17];
  //     snprintf(buf, 17, "Hold: %2us left  ", remaining);
  //     lcd.setCursor(0, 1);
  //     lcd.print(buf);
  //   }
  // }

  // ----- Timeouts (Pi unresponsive or script not running) -----
  if (waitingForScan && now - scanSentTime > SCAN_TIMEOUT_MS) {
    waitingForScan = false;
    lcdPrint("Scan timed out  ", "Is Pi running?  ");
  }

  if (waitingForCapture && now - captureSentTime > CAPTURE_TIMEOUT_MS) {
    waitingForCapture = false;
    char buf[17];
    snprintf(buf, 17, "Frame %2d/20     ", calibFrameCount);
    lcdPrint(buf, "Pi not responding");
  }

  // ----- Messages from Pi -----
  if (!Serial.available()) return;

  String msg = Serial.readStringUntil('\n');
  msg.trim();
  if (msg.length() == 0) return;

  if (msg == "SYSTEM_BOOT") {
    waitingForScan = false;
    waitingForCapture = false;
    lcdPrint("Starting Pi...  ", "Please wait...  ");

  } else if (msg == "WAIT_CAMERAS") {
    waitingForScan = false;
    waitingForCapture = false;
    lcdPrint("No Cameras Found", "Check USB cable ");

  } else if (msg == "WAIT_PROJECTOR") {
    waitingForScan = false;
    waitingForCapture = false;
    lcdPrint("Projector Off   ", "Check DLP power ");

  } else if (msg == "CAMERA_ERR") {
    waitingForScan = false;
    waitingForCapture = false;
    lcdPrint("Camera Error!   ", "Reconnecting... ");

  } else if (msg == "SYSTEM_RESTART") {
    waitingForScan = false;
    waitingForCapture = false;
    calibModeActive = false;
    lcdPrint("Restarting Pi   ", "Please wait...  ");

  } else if (msg == "SCANNING") {
    scanSentTime = now;
    resultDisplayed = false;
    lcdPrint("Scanning...     ", "Hold still...   ");

  } else if (msg == "RECONSTRUCTING") {
    scanSentTime = now;
    lcdPrint("Reconstructing  ", "Please wait...  ");

  } else if (msg == "ROUGHNESS") {
    scanSentTime = now;
    lcdPrint("Analyzing       ", "roughness...    ");

  } else if (msg.startsWith("RESULT ")) {
    waitingForScan = false;
    showResult(msg);

  } else if (msg == "ANALYSIS_FAIL") {
    waitingForScan = false;
    lcdPrint("Analysis failed ", "Check Pi log    ");

  } else if (msg == "SCAN_INCOMPLETE") {
    waitingForScan = false;
    lcdPrint("Scan incomplete ", "Try again       ");

  } else if (msg == "SCAN_DONE") {
    waitingForScan = false;
    if (!resultDisplayed) {
      showIdle();
    }

  } else if (msg == "CALIB_MODE") {
    waitingForScan   = false;
    calibModeActive  = true;
    calibFrameCount  = 0;
    lcdPrint("Calibration mode", "Aim at checker  ");

  } else if (msg.startsWith("FRAME ") && msg.indexOf('/') > 0) {
    waitingForCapture = false;
    int spaceIdx      = msg.indexOf(' ');
    int slashIdx      = msg.indexOf('/');
    calibFrameCount   = msg.substring(spaceIdx + 1, slashIdx).toInt();
    char line0[17];
    snprintf(line0, 17, "Frame %2d/20  OK ", calibFrameCount);
    const char* hint  = calibFrameCount >= MIN_CALIB_FRAMES
                        ? "Flip off=Save   "
                        : "Keep capturing  ";
    lcdPrint(line0, hint);

  } else if (msg == "FRAME ERR") {
    waitingForCapture = false;
    char line0[17];
    snprintf(line0, 17, "Frame %2d/20     ", calibFrameCount);
    lcdPrint(line0, "Capture failed  ");

  } else if (msg == "FRAME BAD") {
    waitingForCapture = false;
    char line0[17];
    snprintf(line0, 17, "Frame %2d/20     ", calibFrameCount);
    lcdPrint(line0, "No corners found");

  } else if (msg == "CALIB_COMPUTING") {
    lcdPrint("Computing...    ", "Please wait...  ");

  } else if (msg == "CALIB_DONE") {
    lcdPrint("Calib saved!    ", "Returning...    ");

  } else if (msg == "CALIB_FAIL") {
    lcdPrint("Calib failed    ", "Too few frames  ");

  } else if (msg == "CALIB_REJECTED") {
    lcdPrint("Calib rejected  ", "Old one kept    ");

  } else if (msg == "SCAN_MODE") {
    waitingForCapture = false;
    calibModeActive   = false;
    calibFrameCount   = 0;
    showIdle();
  }
}
