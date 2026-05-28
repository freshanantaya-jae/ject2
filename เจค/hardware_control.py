import time

# Try to import PYNQ library (available on the physical PYNQ-Z2 board)
try:
    from pynq.overlays.base import BaseOverlay
    from pynq import GPIO
    HAS_PYNQ = True
except ImportError:
    HAS_PYNQ = False

class HardwareController:
    def __init__(self, buzzer_pin_index=0):
        """Initializes the PYNQ-Z2 hardware controller.
        
        Args:
            buzzer_pin_index (int): Pin index for the Buzzer on PMOD or Arduino.
                                    In PYNQ, we can initialize GPIO directly.
        """
        self.has_pynq = HAS_PYNQ
        self.base = None
        self.buzzer = None
        
        if self.has_pynq:
            try:
                # Load the base overlay to enable onboard LEDs, Buttons, etc.
                self.base = BaseOverlay("base.bit")
                print("PYNQ Base Overlay loaded successfully.")
                
                # Set up Buzzer GPIO pin (e.g. PMOD A Pin 1 corresponds to a specific Zynq GPIO)
                # For simplicity, we can use the PYNQ GPIO class.
                # Adjust gpio_pin number based on connection (e.g., PMOD A pin 1 is typically index 0 in PMOD GPIO)
                self.buzzer = GPIO(GPIO.get_gpio_pin(buzzer_pin_index), 'out')
                print(f"Buzzer initialized on GPIO pin index {buzzer_pin_index}")
            except Exception as e:
                print(f"Error loading PYNQ Hardware: {e}. Switching to Mock Mode.")
                self.has_pynq = False
        else:
            print("PYNQ library not detected. Running in Hardware Simulation (Mock) Mode.")

    def set_led(self, led_index, state):
        """Sets the state of an onboard LED (0 to 3).
        
        Args:
            led_index (int): LED index (0-3)
            state (bool): True to turn ON, False to turn OFF
        """
        if self.has_pynq and self.base:
            try:
                if 0 <= led_index < len(self.base.leds):
                    if state:
                        self.base.leds[led_index].on()
                    else:
                        self.base.leds[led_index].off()
            except Exception as e:
                print(f"Failed to set LED {led_index}: {e}")
        else:
            status = "ON" if state else "OFF"
            # print(f"[MOCK HARDWARE] LED {led_index} set to {status}")

    def set_rgb_led(self, rgb_index, color_code):
        """Sets the color of onboard RGB LEDs (4 or 5).
        
        Args:
            rgb_index (int): RGB LED index (4 or 5)
            color_code (int): 0 (off), 1 (blue), 2 (green), 3 (cyan), 
                              4 (red), 5 (magenta), 6 (yellow), 7 (white)
        """
        if self.has_pynq and self.base:
            try:
                # In PYNQ, rgbleds are indexed 4 and 5 (or 0 and 1 depending on base overlay)
                # base.rgbleds[0] is LD4, base.rgbleds[1] is LD5
                idx = rgb_index - 4 if rgb_index >= 4 else rgb_index
                if 0 <= idx < len(self.base.rgbleds):
                    self.base.rgbleds[idx].write(color_code)
            except Exception as e:
                print(f"Failed to set RGB LED {rgb_index}: {e}")
        else:
            colors = ["OFF", "BLUE", "GREEN", "CYAN", "RED", "MAGENTA", "YELLOW", "WHITE"]
            color_name = colors[color_code] if 0 <= color_code < 8 else "UNKNOWN"
            # print(f"[MOCK HARDWARE] RGB LED {rgb_index} set to color: {color_name}")

    def beep(self, duration=0.2):
        """Triggers the buzzer for a specified duration."""
        if self.has_pynq and self.buzzer:
            try:
                self.buzzer.write(1)  # Turn buzzer ON
                time.sleep(duration)
                self.buzzer.write(0)  # Turn buzzer OFF
            except Exception as e:
                print(f"Failed to control Buzzer: {e}")
        else:
            print(f"[MOCK HARDWARE] BUZZER BEEP for {duration} seconds (BEEP! BEEP!)")

    def signal_pass(self):
        """Triggers the success signal (Green LEDs, short single beep)."""
        print("Hardware signal: PASS")
        # Turn on green LEDs
        self.set_led(0, True)
        self.set_led(1, True)
        self.set_rgb_led(4, 2)  # Green on LD4
        
        # Single short beep
        self.beep(0.1)
        
        # Let them stay on for a moment, then reset
        time.sleep(0.5)
        self.set_led(0, False)
        self.set_led(1, False)
        self.set_rgb_led(4, 0)

    def signal_fail(self):
        """Triggers the failure signal (Red LEDs, 3 long beeps)."""
        print("Hardware signal: FAIL")
        # Turn on red LEDs
        self.set_led(2, True)
        self.set_led(3, True)
        self.set_rgb_led(4, 4)  # Red on LD4
        
        # 3 quick alert beeps
        for _ in range(3):
            self.beep(0.15)
            time.sleep(0.1)
            
        time.sleep(0.3)
        self.set_led(2, False)
        self.set_led(3, False)
        self.set_rgb_led(4, 0)

if __name__ == '__main__':
    # Test Hardware Controller
    controller = HardwareController()
    
    print("Testing SUCCESS signal:")
    controller.signal_pass()
    
    time.sleep(1)
    
    print("Testing FAILURE signal:")
    controller.signal_fail()
