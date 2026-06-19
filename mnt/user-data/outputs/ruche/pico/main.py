# ============================================================
#  Ruche Dashboard — Firmware Raspberry Pi Pico WH
#  MicroPython >= 1.22
# ============================================================
#
#  Capteurs supportés :
#    - DHT22 (température + humidité) sur GP15
#    - Tension batterie via ADC sur GP28 (A2) avec pont diviseur
#    - Comptage abeilles : simulé ici, à brancher sur ton détecteur
#      (barrière IR, cellule optique, …)
#
#  Fichier de config : config.json (dans le flash de la Pico)
#
#  Câblage DHT22 :
#    VCC  → 3.3 V
#    DATA → GP15 + résistance pull-up 10 kΩ vers 3.3 V
#    GND  → GND
#
#  Câblage batterie (pont diviseur 1:2) :
#    VBAT → R1(100k) → GP28 / A2 → R2(100k) → GND
#    (adapte les résistances selon ta tension batterie max)
# ============================================================

import json
import time
import math
import network
import urequests
import machine
from machine import ADC, Pin

# ── Chargement config ────────────────────────────────────────
def load_config():
    try:
        with open("config.json") as f:
            return json.load(f)
    except Exception as e:
        print("[config] ERREUR lecture config.json :", e)
        raise SystemExit("Pas de config.json — impossible de démarrer.")

CFG = load_config()

WIFI_SSID     = CFG["wifi_ssid"]
WIFI_PASS     = CFG["wifi_password"]
BACKEND_URL   = CFG["backend_url"]        # ex: https://ruche-api.onrender.com
DEVICE_ID     = CFG.get("device_id", "ruche-1-pico")
RUCHE_TOKEN   = CFG.get("ruche_token", "")  # token secret, laisser "" si pas d'auth
SEND_INTERVAL = CFG.get("send_interval_s", 30)  # secondes entre chaque envoi

# ── LED onboard ──────────────────────────────────────────────
led = machine.Pin("LED", machine.Pin.OUT)

def blink(n=1, t=0.1):
    for _ in range(n):
        led.on();  time.sleep(t)
        led.off(); time.sleep(t)

# ── Wi-Fi ─────────────────────────────────────────────────────
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return wlan
    print(f"[wifi] Connexion à {WIFI_SSID} …")
    wlan.connect(WIFI_SSID, WIFI_PASS)
    timeout = 20
    while not wlan.isconnected() and timeout > 0:
        time.sleep(1)
        timeout -= 1
        led.toggle()
    led.off()
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"[wifi] Connecté — IP: {ip}")
        blink(3, 0.08)
        return wlan
    else:
        print("[wifi] ECHEC connexion Wi-Fi")
        return None

# ── DHT22 ────────────────────────────────────────────────────
try:
    import dht
    _dht = dht.DHT22(Pin(15))
    HAS_DHT = True
    print("[capteur] DHT22 initialisé sur GP15")
except ImportError:
    HAS_DHT = False
    print("[capteur] DHT22 non disponible (module dht absent)")

def read_dht():
    if not HAS_DHT:
        # Simulation avec légère dérive pour tester
        t = 22.0 + math.sin(time.time() / 600) * 4
        h = 58.0 + math.cos(time.time() / 400) * 10
        return round(t, 1), round(h, 1)
    try:
        _dht.measure()
        time.sleep_ms(100)
        return _dht.temperature(), _dht.humidity()
    except Exception as e:
        print("[dht] Erreur lecture :", e)
        return None, None

# ── Batterie (ADC) ────────────────────────────────────────────
# Pont diviseur 1:2 → Vpico_max = 1.65 V correspond à Vbat_max
# Modifie VBAT_MAX selon ta batterie (ex: 4.2 V lipo, 6 V NiMH…)
VBAT_MAX = CFG.get("vbat_max_v", 4.2)   # tension batterie pleine
VBAT_MIN = CFG.get("vbat_min_v", 3.0)   # tension batterie vide

_adc_batt = ADC(Pin(28))

def read_battery_percent():
    raw    = _adc_batt.read_u16()          # 0 – 65535
    v_adc  = raw * 3.3 / 65535            # tension sur la pin ADC
    v_bat  = v_adc * 2                    # pont diviseur 1:2
    pct    = (v_bat - VBAT_MIN) / (VBAT_MAX - VBAT_MIN) * 100
    return max(0, min(100, round(pct, 1)))

# ── Compteur abeilles (placeholder) ──────────────────────────
# À remplacer par ton vrai capteur :
#   - Deux barrières IR (entrée / sortie) sur des GPIO
#   - Module de vision externe (cam USB + serveur)
#   - Compteur impulsionnel sur interruption

_bee_in  = 0
_bee_out = 0

def read_bee_counts():
    """
    SIMULATION : remplace ce corps par la lecture de tes vrais capteurs.

    Exemple avec deux IRQ sur GP2 (entrée) et GP3 (sortie) :

        _pin_in  = Pin(2, Pin.IN, Pin.PULL_UP)
        _pin_out = Pin(3, Pin.IN, Pin.PULL_UP)
        def _irq_in(p):  global _bee_in;  _bee_in  += 1
        def _irq_out(p): global _bee_out; _bee_out += 1
        _pin_in.irq( trigger=Pin.IRQ_FALLING, handler=_irq_in)
        _pin_out.irq(trigger=Pin.IRQ_FALLING, handler=_irq_out)
    """
    global _bee_in, _bee_out
    # Simulation : flux aléatoire pendant les heures "actives"
    h = (time.localtime()[3] + 1) % 24  # heure UTC+1
    if 8 <= h <= 18:
        import random
        _bee_in  += random.randint(0, 5)
        _bee_out += random.randint(0, 4)
    return _bee_in, _bee_out

# ── Uptime ───────────────────────────────────────────────────
_boot_ms = time.ticks_ms()

def uptime_s():
    return time.ticks_diff(time.ticks_ms(), _boot_ms) // 1000

# ── Envoi au backend ──────────────────────────────────────────
def send_data(wlan):
    temp, hum = read_dht()
    batt      = read_battery_percent()
    b_in, b_out = read_bee_counts()
    up        = uptime_s()

    payload = {
        "device_id":        DEVICE_ID,
        "temperature_c":    temp,
        "humidity_percent": hum,
        "battery_percent":  batt,
        "bee_in":           b_in,
        "bee_out":          b_out,
        "uptime_s":         up,
    }

    headers = {
        "Content-Type":  "application/json",
        "User-Agent":    "RuchePico/1.0 MicroPython",
    }
    if RUCHE_TOKEN:
        headers["X-Ruche-Token"] = RUCHE_TOKEN

    url = BACKEND_URL.rstrip("/") + "/api/pico"
    print(f"[send] → {url}")
    print(f"        T={temp}°C H={hum}% batt={batt}% in={b_in} out={b_out} uptime={up}s")

    try:
        r = urequests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        print(f"[send] ← {r.status_code} {r.text[:80]}")
        r.close()
        blink(1, 0.05)
        return True
    except Exception as e:
        print(f"[send] ERREUR : {e}")
        blink(5, 0.05)  # 5 clignotements rapides = erreur réseau
        return False

# ── Reconnexion auto ──────────────────────────────────────────
def ensure_wifi(wlan):
    if wlan and wlan.isconnected():
        return wlan
    print("[wifi] Reconnexion…")
    return connect_wifi()

# ── Boucle principale ─────────────────────────────────────────
def main():
    print("=" * 48)
    print(f"  Ruche Dashboard — Pico WH Firmware v1.0")
    print(f"  Device  : {DEVICE_ID}")
    print(f"  Backend : {BACKEND_URL}")
    print(f"  Interval: {SEND_INTERVAL} s")
    print("=" * 48)

    wlan = connect_wifi()
    if not wlan:
        # Pas de Wi-Fi au boot → attendre et retenter indéfiniment
        while True:
            time.sleep(10)
            wlan = connect_wifi()
            if wlan:
                break

    while True:
        wlan = ensure_wifi(wlan)
        if wlan:
            send_data(wlan)
        else:
            print("[loop] Wi-Fi indisponible, attente 30 s…")

        # Attendre l'intervalle (avec vérification régulière)
        for _ in range(SEND_INTERVAL):
            time.sleep(1)

main()
