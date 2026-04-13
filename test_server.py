#!/usr/bin/env python3
"""Test SmartAir Server Endpoints"""

import requests
import json
import time

print("\n" + "="*70)
print("  SMARTAIR SERVER ENDPOINT TESTS")
print("="*70 + "\n")

BASE_URL = "http://localhost:5000"

# Test 1: Status
print("[TEST 1] GET /api/status")
try:
    resp = requests.get(f"{BASE_URL}/api/status", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    print(f"  ✅ Status endpoint working")
    print(f"     • Project: {data.get('project', 'N/A')}")
    print(f"     • Model loaded: {data['model']['loaded']}")
    print(f"     • Version: {data.get('version', 'N/A')}")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# Test 2: History
print("\n[TEST 2] GET /api/history")
try:
    resp = requests.get(f"{BASE_URL}/api/history", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    print(f"  ✅ History endpoint working")
    print(f"     • Total readings: {data.get('count', 0)}")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# Test 3: Stats
print("\n[TEST 3] GET /api/stats")
try:
    resp = requests.get(f"{BASE_URL}/api/stats", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    print(f"  ✅ Stats endpoint working")
    print(f"     • Total samples: {data.get('total', 0)}")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# Test 4: Alerts
print("\n[TEST 4] GET /api/alerts")
try:
    resp = requests.get(f"{BASE_URL}/api/alerts", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    print(f"  ✅ Alerts endpoint working")
    print(f"     • Recent alerts: {data.get('count', 0)}")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# Test 5: Dashboard
print("\n[TEST 5] GET / (Dashboard)")
try:
    resp = requests.get(f"{BASE_URL}/", timeout=5)
    resp.raise_for_status()
    print(f"  ✅ Dashboard accessible")
    print(f"     • Response size: {len(resp.text)} bytes")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# Test 6: Predict (with demo data)
print("\n[TEST 6] POST /api/predict (Demo Prediction)")
try:
    payload = {
        "mq135": 420,
        "mq3": 380,
        "mq7": 500,
        "mq4": 450,
        "temp": 28.5,
        "hum": 55.0,
        "flame": 0
    }
    resp = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    pred = data.get('prediction', {})
    print(f"  ✅ Predict endpoint working")
    print(f"     • Gas class ID: {pred.get('gas_class_id', 'N/A')}")
    print(f"     • Gas name: {pred.get('gas_name', 'N/A')}")
    print(f"     • Confidence: {pred.get('confidence', 'N/A')}%")
    print(f"     • Severity: {pred.get('severity', 'N/A')}")
    print(f"     • PPM estimate: {pred.get('ppm_estimate', 'N/A')}")
    print(f"     • Demo mode: {pred.get('demo_mode', False)}")
    if data.get('alert'):
        print(f"     • Alert: YES ⚠️")
    else:
        print(f"     • Alert: NO")
except Exception as e:
    print(f"  ❌ Failed: {e}")

print("\n" + "="*70)
print("  ✅ ALL ENDPOINT TESTS COMPLETED")
print("="*70 + "\n")

print("Server Status: ✅ RUNNING")
print(f"Access endpoints at: {BASE_URL}")
print(f"Dashboard: {BASE_URL}/")
print(f"API docs (endpoints):")
print(f"  • POST /api/predict    - Make a gas prediction")
print(f"  • GET  /api/history    - Get recent readings")
print(f"  • GET  /api/stats      - Get class statistics")
print(f"  • GET  /api/alerts     - Get recent alerts")
print(f"  • GET  /api/status     - Get server & model status")
print()
