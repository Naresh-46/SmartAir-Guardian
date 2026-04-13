#!/usr/bin/env python3
"""
SmartAir Guardian — ESP Setup Assistant
Helps configure ESP WiFi connection to local server
"""

import socket
import netifaces
import json
from pathlib import Path

def get_local_ip():
    """Get local machine IPv4 address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"[ERR] Could not detect IP: {e}")
        return None

def get_all_ips():
    """List all available network interfaces"""
    ips = {}
    try:
        for interface in netifaces.interfaces():
            if_addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in if_addrs:
                ipv4 = if_addrs[netifaces.AF_INET][0]['addr']
                if not ipv4.startswith('127.'):  # Skip loopback
                    ips[interface] = ipv4
    except Exception as e:
        print(f"[WARN] Could not enumerate interfaces: {e}")
    return ips

def test_server_connection(ip_addr):
    """Test if Flask server is running"""
    try:
        import requests
        response = requests.get(f"http://{ip_addr}:5000/api/status", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return True, data
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("  SmartAir Guardian — ESP WiFi Setup Assistant")
    print("=" * 60)
    
    # Step 1: Detect IP
    print("\n[1] Local Network Configuration")
    print("-" * 60)
    
    local_ip = get_local_ip()
    if local_ip:
        print(f"✅ Primary IPv4 Address: {local_ip}")
    
    all_ips = get_all_ips()
    if all_ips:
        print(f"\n📡 All Available Networks:")
        for iface, ip in all_ips.items():
            print(f"   • {iface:<15} → {ip}")
    
    # Step 2: Test Flask server
    print("\n[2] Flask Server Status")
    print("-" * 60)
    
    if local_ip:
        is_running, result = test_server_connection(local_ip)
        if is_running:
            print(f"✅ Flask server is RUNNING")
            print(f"   Model: {result.get('model_path', 'unknown')}")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"❌ Flask server is NOT running")
            print(f"   Error: {result}")
            print(f"   Start it with: python -m server.app")
    
    # Step 3: Generate configuration
    print("\n[3] ESP Configuration")
    print("-" * 60)
    
    if local_ip:
        config = {
            "WIFI_SSID": "YOUR_SSID",
            "WIFI_PASSWORD": "YOUR_PASSWORD",
            "SERVER_IP": local_ip,
            "SERVER_PORT": 5000,
            "API_ENDPOINT": "/api/predict"
        }
        
        print("📋 Copy these values to ESP firmware (lines 12-14):")
        print()
        print(f'const char* WIFI_SSID     = "{config["WIFI_SSID"]}";')
        print(f'const char* WIFI_PASSWORD = "{config["WIFI_PASSWORD"]}";')
        print(f'const char* SERVER_IP     = "{config["SERVER_IP"]}";')
        print()
        
        # Save to file
        config_file = Path("esp_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Configuration saved: {config_file}")
    
    # Step 4: Instructions
    print("\n[4] Next Steps")
    print("-" * 60)
    print("1. Edit ESP firmware (smartair_http_client.ino lines 12-14)")
    print("   Replace WiFi SSID and PASSWORD with your network")
    print(f"   Set SERVER_IP = {local_ip}")
    print("")
    print("2. Upload firmware to ESP8266 using Arduino IDE")
    print("   Board: NodeMCU 1.0 (ESP-12E Module)")
    print("   Speed: 115200 baud")
    print("")
    print("3. Start Flask server (if not running)")
    print("   Command: python -m server.app")
    print("")
    print("4. Open Arduino Serial Monitor (115200 baud)")
    print("   Type: START")
    print("")
    print("5. Watch ESP send predictions to ML model!")
    print("")
    print("=" * 60)

if __name__ == "__main__":
    main()
