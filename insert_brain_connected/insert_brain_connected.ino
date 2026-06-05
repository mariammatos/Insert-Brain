// ============================================================
// FILE: insert_brain_connected.ino
//
// Firmware for robot control via BCI (Insert-Brain).
//
// Accepts two serial protocols (9600 baud):
//
// 1. BCI PROTOCOL (from test_model_connected.py):
//    LEFT   → active servo moves backward (-stepSize)
//    RIGHT  → active servo moves forward (+stepSize)
//    FEET   → switch active servo (0→1→2→3→0→...)
//    REST   → no movement
//    STOP   → move all servos to neutral position
//
// 2. MANUAL PROTOCOL (compatible with insert_brain.ino):
//    +1 -1 +2 -2 +3 -3 +4 -4
//    Repeated digit = multiple steps (e.g. +111 = 3 steps)
//
// Always replies with a Serial confirmation.
// ============================================================

#include <Servo.h>

Servo Servo_0;
Servo Servo_1;
Servo Servo_2;
Servo Servo_3;

// Current positions
int M0 = 90, M1 = 90, M2 = 90, M3 = 160;

// Angle limits
int minAngle[4]    = {10,  10,  10,  100};
int maxAngle[4]    = {170, 170, 170, 170};
int neutralAngle[4] = {90,  90,  90,  160};

// Degrees per command step
int stepSize = 5;

// Servo currently selected by BCI (index 0-3)
int activeServo = 0;

bool clenchDisabled = true;

void setup()
{
  Serial.begin(9600);

  Servo_0.attach(4);
  Servo_1.attach(5);
  Servo_2.attach(6);
  Servo_3.attach(7);

  Servo_0.write(M0);
  Servo_1.write(M1);
  Servo_2.write(M2);
  Servo_3.write(M3);

  Serial.println("INSERT-BRAIN ready.");
  Serial.print("Active servo: ");
  Serial.println(activeServo + 1);
  Serial.println("BCI: LEFT RIGHT FEET REST STOP");
  Serial.println("Manual: +1 -1 +2 -2 +3 -3 +4 -4");
}


void loop()
{
  if (Serial.available() > 0)
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) return;
    processCommand(cmd);
  }
}


void processCommand(String cmd)
{
  // ── BCI PROTOCOL ──────────────────────────────────────────

  if (cmd == "LEFT")
  {
    Serial.print("BCI: LEFT -> servo ");
    Serial.print(activeServo + 1);
    Serial.println(" moves backward");
    moveServo(activeServo, -stepSize);
    return;
  }

  if (cmd == "RIGHT")
  {
    Serial.print("BCI: RIGHT -> servo ");
    Serial.print(activeServo + 1);
    Serial.println(" moves forward");
    moveServo(activeServo, +stepSize);
    return;
  }

  if (cmd == "FEET")
  {
    // Cycle to the next servo
    activeServo = (activeServo + 1) % (clenchDisabled ? 3 : 4);
    Serial.print("BCI: FEET -> active servo is now ");
    Serial.println(activeServo + 1);
    return;
  }

  if (cmd == "REST")
  {
    Serial.println("BCI: REST -> no movement");
    return;
  }

  if (cmd == "STOP")
  {
    Serial.println("BCI: STOP -> neutral position");
    goNeutral();
    return;
  }

  // ── MANUAL PROTOCOL ──────────────────────────────────────

  if (cmd.length() < 2)
  {
    Serial.println("Invalid command.");
    return;
  }

  char dir = cmd.charAt(0);
  if (dir != '+' && dir != '-')
  {
    Serial.println("Must start with + or -");
    return;
  }

  char servoChar = cmd.charAt(1);
  int  servoNum  = servoChar - '0';

  if (servoNum < 1 || servoNum > 4)
  {
    Serial.println("Servo must be 1-4");
    return;
  }

  int steps = 0;
  for (int i = 1; i < (int)cmd.length(); i++)
  {
    if (cmd.charAt(i) == servoChar) steps++;
    else break;
  }

  int delta = (dir == '+') ? stepSize * steps : -stepSize * steps;
  moveServo(servoNum - 1, delta);
}


void moveServo(int idx, int delta)
{
  Servo* servos[4] = {&Servo_0, &Servo_1, &Servo_2, &Servo_3};
  int*   angles[4] = {&M0, &M1, &M2, &M3};

  int newAngle = *angles[idx] + delta;
  newAngle = constrain(newAngle, minAngle[idx], maxAngle[idx]);

  int current = *angles[idx];
  int step    = (newAngle > current) ? 1 : -1;
  for (; current != newAngle; current += step)
  {
    servos[idx]->write(current);
    delay(2);
  }

  *angles[idx] = newAngle;

  Serial.print("Servo ");
  Serial.print(idx + 1);
  Serial.print(" -> ");
  Serial.println(newAngle);
}


void goNeutral()
{
  Servo* servos[4] = {&Servo_0, &Servo_1, &Servo_2, &Servo_3};
  int*   angles[4] = {&M0, &M1, &M2, &M3};

  for (int i = 0; i < 4; i++)
  {
    int target  = neutralAngle[i];
    int current = *angles[i];
    int step    = (target > current) ? 1 : -1;
    for (; current != target; current += step)
    {
      servos[i]->write(current);
      delay(2);
    }
    *angles[i] = target;
  }

  activeServo = 0;
  Serial.println("All servos -> neutral position");
  Serial.println("Active servo reset to 1");
}
