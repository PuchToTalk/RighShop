import subprocess
import time

def detect_adb_device():
    """wait for Android device (set-top box) to be connected via adb."""
    while True:
        try:
            # Check for connected devices
            result = subprocess.run(['/usr/bin/adb', 'devices'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout.splitlines()
            devices = [line.split('\t')[0] for line in output if '\tdevice' in line]
            
            if devices:
                print("Device detected")
                subprocess.run(['/usr/bin/adb', 'root'])
                return True
            else:
                print("Waiting for device...")
                time.sleep(2)  # check every 2 sec
        
        except subprocess.CalledProcessError as e:
            print(f"Error running adb devices: {e}")
            time.sleep(2)  # wait 2 sec before retrying

if __name__ == "__main__":
    detect_adb_device()