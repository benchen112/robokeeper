#include <SoftwareSerial.h>

// IMPORTANT: Do NOT use Pins 0 and 1 (they are shared with USB Serial).
// Connect Arduino Pin 3 (TX) -> 5V-to-3.3V Level Shifter -> Pico GP1 (Pin 2)
// Connect Arduino GND -> Pico GND
SoftwareSerial picoSerial(2, 3); // RX = Pin 2, TX = Pin 3

void setup() {
  // USB Serial Monitor stays at 115200 baud
  Serial.begin(115200);
  while (!Serial);

  // Use 9600 baud for SoftwareSerial so AVR bit-banging doesn't corrupt bytes
  picoSerial.begin(9600);

  Serial.println("=== TOSS UART Angle Tester ===");
  Serial.println("Enter target goalie angle in DEGREES (-90 to +90):");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() > 0) {
      float angle_deg = input.toFloat();
      
      // Convert degrees to radians to match Pico's strtof() expectation
      float angle_rad = angle_deg * (PI / 180.0);

      // Send clean float in radians followed by '\n' out of Pin 3
      picoSerial.print(angle_rad, 4);
      picoSerial.print('\n');

      // Print debug message to USB Serial Monitor (Pins 0/1)
      Serial.print("Typed: ");
      Serial.print(angle_deg, 1);
      Serial.print(" deg  -->  Sent to Pico: ");
      Serial.print(angle_rad, 4);
      Serial.println(" rad");
    }
  }
}