import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import pyautogui
import requests
import time
import os
import sys
import threading
import re
import subprocess
import math

from PIL import Image
import base64
from io import BytesIO

import hashlib
import hmac
import json
import uuid

from pynput import keyboard

# ================= CONFIG =================
APP_NAME = "OCRBot"

APP_SUPPORT_DIR = os.path.join(os.path.expanduser("~/Library/Application Support"), APP_NAME)
CONFIG_FILE = os.path.join(APP_SUPPORT_DIR, "ocr_bot_api.txt")
LOG_FILE = os.path.join(APP_SUPPORT_DIR, "ocr_bot_log.txt")
LICENSE_FILE = os.path.join(APP_SUPPORT_DIR, "license.json")

PRODUCT_ID = "OCRBot-v1"
LIC_SECRET_B64 = "VMpqfFouIuDX_CNDNKDhxvYwCZxLUtiVDVt2GGdU-ks="

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

MIN_INTERVAL_F2 = 3.0
MIN_INTERVAL_F4 = 3.0
MIN_INTERVAL_F6 = 3.0
MIN_INTERVAL_F5 = 1.2

PIXELS = 20
DELAY = 0.4

MOVE_OUT_DURATION = 0.22
MOVE_BACK_DURATION = 0.18
MOVE_TWEEN = pyautogui.easeInOutQuad
STEP_PAUSE = 1.0

HEX_R = 24
GEMINI_MIN_GAP = 15

VISION_MAX_W = 1280
VISION_JPEG_QUALITY = 70

pyautogui.FAILSAFE = True
# =========================================


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def log(msg: str):
    try:
        _ensure_dir(LOG_FILE)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def log_block(tag: str, text: str):
    try:
        _ensure_dir(LOG_FILE)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tag}\n")
            f.write("-" * 80 + "\n")
            f.write((text or "") + "\n")
            f.write("=" * 80 + "\n")
    except Exception as e:
        log(f"log_block error: {repr(e)}")


def take_center_screenshot():
    w, h = pyautogui.size()
    top = int(h * 0.10)
    height = int(h * 0.80)
    return pyautogui.screenshot(region=(0, top, w, height))


def _prepare_vision_image(pil_image: Image.Image) -> (str, str):
    img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
    if img.width > VISION_MAX_W:
        scale = VISION_MAX_W / float(img.width)
        img = img.resize((VISION_MAX_W, max(1, int(round(img.height * scale)))))
    bio = BytesIO()
    img.save(bio, format="JPEG", quality=VISION_JPEG_QUALITY, optimize=True)
    return "image/jpeg", base64.b64encode(bio.getvalue()).decode("ascii")


def _do_gemini_call_vision(model_name: str, text_prompt: str, mime_type: str, img_b64: str, api_key: str) -> requests.Response:
    url = "https://generativelanguage.googleapis.com/v1beta/" f"models/{model_name}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": text_prompt}, {"inline_data": {"mime_type": mime_type, "data": img_b64}}]}]}
    return requests.post(url, json=payload, timeout=35)


_last_gemini_call_ts = 0.0
_gemini_lock = threading.Lock()


def _gemini_request_vision(pil_image: Image.Image, text_prompt: str, api_key: str) -> str:
    global _last_gemini_call_ts
    mime_type, img_b64 = _prepare_vision_image(pil_image)

    with _gemini_lock:
        now = time.time()
        gap = now - _last_gemini_call_ts
        if gap < GEMINI_MIN_GAP:
            time.sleep(GEMINI_MIN_GAP - gap)

        backoff = 1.0
        max_tries_per_model = 2

        for model in GEMINI_MODELS:
            for _ in range(max_tries_per_model):
                _last_gemini_call_ts = time.time()
                try:
                    r = _do_gemini_call_vision(model, text_prompt, mime_type, img_b64, api_key)
                except requests.RequestException as e:
                    log(f"Gemini network error ({model}): {repr(e)}")
                    time.sleep(min(backoff, 6.0))
                    backoff = min(backoff * 2, 10.0)
                    continue

                if r.status_code == 404:
                    log(f"Gemini model not available: {model}")
                    break
                if r.status_code == 429:
                    log(f"Gemini 429 on {model}")
                    break
                if 500 <= r.status_code <= 599:
                    time.sleep(min(backoff, 8.0))
                    backoff = min(backoff * 2, 10.0)
                    continue

                r.raise_for_status()
                try:
                    data = r.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception as e:
                    log(f"Gemini parse error: {repr(e)}")
                    return ""

            log("Switching Gemini model -> next")

        raise RuntimeError("All Gemini models failed")


def normalize_answer_ad(txt: str):
    if not txt:
        return None
    m = re.search(r"\b([ABCD])\b", txt.strip().upper())
    return m.group(1) if m else None


def normalize_multi_answers_af(txt: str):
    if not txt:
        return []
    found = re.findall(r"\b([A-F])\b", txt.upper())
    out = []
    for c in ("A", "B", "C", "D", "E", "F"):
        if c in found and c not in out:
            out.append(c)
    return out


def parse_steps_1to8_unique(text: str, max_n: int = 8):
    if not text:
        return None
    nums, cur = [], ""
    for ch in text:
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                nums.append(int(cur))
                cur = ""
    if cur:
        nums.append(int(cur))
    if not nums:
        return None

    out = []
    for x in nums:
        if not (1 <= x <= 8):
            return None
        if x not in out:
            out.append(x)
        if len(out) >= max_n:
            break
    return out or None


def move_mouse(answer: str):
    if answer == "A":
        pyautogui.moveRel(0, -PIXELS)
    elif answer == "B":
        pyautogui.moveRel(PIXELS, 0)
    elif answer == "C":
        pyautogui.moveRel(0, PIXELS)
    elif answer == "D":
        pyautogui.moveRel(-PIXELS, 0)


def move_mouse_8_out_and_back(step: int):
    dirs = {
        1: (0, -PIXELS),
        2: (PIXELS, -PIXELS),
        3: (PIXELS, 0),
        4: (PIXELS, PIXELS),
        5: (0, PIXELS),
        6: (-PIXELS, PIXELS),
        7: (-PIXELS, 0),
        8: (-PIXELS, -PIXELS),
    }
    dxdy = dirs.get(step)
    if not dxdy:
        return
    dx, dy = dxdy
    pyautogui.moveRel(dx, dy, duration=MOVE_OUT_DURATION, tween=MOVE_TWEEN)
    time.sleep(STEP_PAUSE)
    pyautogui.moveRel(-dx, -dy, duration=MOVE_BACK_DURATION, tween=MOVE_TWEEN)


def move_mouse_deg_out_and_back(deg: float, radius: int = HEX_R):
    th = math.radians(deg)
    dx = int(round(radius * math.sin(th)))
    dy = int(round(-radius * math.cos(th)))
    pyautogui.moveRel(dx, dy, duration=MOVE_OUT_DURATION, tween=MOVE_TWEEN)
    time.sleep(STEP_PAUSE)
    pyautogui.moveRel(-dx, -dy, duration=MOVE_BACK_DURATION, tween=MOVE_TWEEN)


# ===== Tk-thread-safe helpers =====
root = None
status_var = None

def set_status(text: str):
    try:
        root.after(0, status_var.set, text)
    except Exception:
        pass


def _set_clipboard_on_main(text: str):
    # Runs ONLY on Tk main thread
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update_idletasks()


def paste_at_cursor_threadsafe(text: str):
    def do_all():
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update_idletasks()
        time.sleep(0.05)
        pyautogui.hotkey("command", "v")  # ВАЖНО: теперь в main thread

    try:
        run_on_main(do_all, timeout=3.0)
    except Exception as e:
        log(f"paste error: {repr(e)}")


# ===== LICENSE =====
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(s + pad)

def _secret_bytes() -> bytes:
    return _b64url_decode(LIC_SECRET_B64)

def _mac_hardware_uuid() -> str:
    try:
        out = subprocess.check_output(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', out)
        if m:
            return m.group(1).strip().lower()
    except Exception:
        pass
    return ""

def get_machine_id() -> str:
    hw = _mac_hardware_uuid()
    return hw if hw else str(uuid.getnode()).strip().lower()

def sign_payload(payload: dict) -> str:
    secret = _secret_bytes()
    msg = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(secret, msg, hashlib.sha256).digest()
    return _b64url(sig)

def parse_license_string(lic: str):
    try:
        body_b64, sig = lic.strip().split(".", 1)
        payload = json.loads(_b64url_decode(body_b64).decode("utf-8"))
        return payload, sig
    except Exception:
        return None, None

def verify_license_string(lic: str) -> (bool, str):
    payload, sig = parse_license_string(lic)
    if not payload or not sig:
        return False, "Bad license format"

    expected = sign_payload(payload)
    if not hmac.compare_digest(expected, sig):
        return False, "Invalid signature"

    if payload.get("product") != PRODUCT_ID:
        return False, "Wrong product"

    if payload.get("machine") != get_machine_id():
        return False, "License is for another Mac"

    exp = payload.get("exp", 0)
    if isinstance(exp, (int, float)) and float(exp) > 0:
        if time.time() > float(exp):
            return False, "License expired"

    return True, "OK"

def load_saved_license() -> str:
    try:
        if os.path.exists(LICENSE_FILE):
            with open(LICENSE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (data.get("license") or "").strip()
    except Exception:
        pass
    return ""

def save_license(lic: str):
    _ensure_dir(LICENSE_FILE)
    with open(LICENSE_FILE, "w", encoding="utf-8") as f:
        json.dump({"license": lic.strip()}, f, ensure_ascii=False, indent=2)

def show_activation_dialog(root_, reason: str):
    win = tk.Toplevel(root_)
    win.title("Activate OCRBot")
    win.geometry("540x280")
    win.resizable(False, False)
    win.grab_set()

    mid = get_machine_id()

    tk.Label(win, text="Activation required", font=("Segoe UI", 12, "bold")).pack(pady=(12, 4))
    tk.Label(win, text=f"Reason: {reason}", font=("Segoe UI", 9)).pack(pady=(0, 8))

    frm = tk.Frame(win)
    frm.pack(fill="x", padx=12)

    tk.Label(frm, text="Machine ID (send this to seller):", font=("Segoe UI", 9, "bold")).pack(anchor="w")
    mid_entry = tk.Entry(frm)
    mid_entry.pack(fill="x", pady=(2, 8))
    mid_entry.insert(0, mid)
    mid_entry.configure(state="readonly")

    def copy_mid():
        win.clipboard_clear()
        win.clipboard_append(mid)
        win.update()

    tk.Button(frm, text="Copy Machine ID", command=copy_mid).pack(anchor="w", pady=(0, 10))

    tk.Label(frm, text="Paste license key here:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
    lic_entry = tk.Entry(frm)
    lic_entry.pack(fill="x", pady=(2, 8))

    status = tk.StringVar(value="")

    def activate():
        lic = lic_entry.get().strip()
        ok, msg = verify_license_string(lic)
        if not ok:
            status.set(msg)
            return
        save_license(lic)
        messagebox.showinfo("License", "Activated!")
        win.destroy()

    tk.Button(win, text="Activate", command=activate).pack(pady=6)
    tk.Label(win, textvariable=status, fg="red", font=("Segoe UI", 9)).pack(pady=(0, 6))

def require_valid_license_or_exit(root_) -> bool:
    lic = load_saved_license()
    ok, msg = verify_license_string(lic) if lic else (False, "No license")
    if ok:
        return True
    show_activation_dialog(root_, msg)
    root_.wait_window(root_.winfo_children()[-1])
    lic2 = load_saved_license()
    ok2, msg2 = verify_license_string(lic2) if lic2 else (False, "No license")
    if not ok2:
        messagebox.showerror("License", f"License not valid: {msg2}")
        return False
    return True


def save_api(key: str):
    _ensure_dir(CONFIG_FILE)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(key)

def load_api() -> str:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


api_key = None
_busy_lock = threading.Lock()
_stop_event = threading.Event()

_last_f2_ts = 0.0
_last_f4_ts = 0.0
_last_f5_ts = 0.0
_last_f6_ts = 0.0

_window_visible = True
_hotkey_listener = None


def action_f2():
    global _last_f2_ts
    now = time.time()
    if now - _last_f2_ts < MIN_INTERVAL_F2:
        return
    _last_f2_ts = now
    if not _busy_lock.acquire(blocking=False):
        return
    try:
        img = take_center_screenshot()
        prompt = "Solve the multiple-choice question in the image.\nReturn ONLY ONE LETTER: A, B, C, or D.\nNo other text."
        raw = _gemini_request_vision(img, prompt, api_key)
        log_block("F2 VISION RAW", raw)
        ans = normalize_answer_ad(raw)
        if ans not in ("A", "B", "C", "D"):
            set_status("HK2: invalid (need A-D)")
            return
        time.sleep(DELAY)
        move_mouse(ans)
        set_status(f"HK2 OK -> {ans}")
    except Exception as e:
        log(f"HK2 ERROR: {repr(e)}")
        set_status(f"HK2 ERROR: {type(e).__name__}")
    finally:
        _busy_lock.release()


def action_f4():
    global _last_f4_ts
    now = time.time()
    if now - _last_f4_ts < MIN_INTERVAL_F4:
        return
    _last_f4_ts = now
    if not _busy_lock.acquire(blocking=False):
        return
    try:
        img = take_center_screenshot()
        prompt = (
            "You are an expert tutor.\nRead the task in the image.\nProduce a COMPLETE final answer.\n"
            "Follow constraints EXACTLY.\nOutput only the final deliverable.\n"
        )
        answer = _gemini_request_vision(img, prompt, api_key)
        log_block("F4 VISION ANSWER", answer)
        if not answer.strip():
            set_status("HK4: empty answer")
            return
        time.sleep(0.15)
        paste_at_cursor_threadsafe(answer)  # ✅ thread-safe now
        set_status("HK4 OK pasted")
    except Exception as e:
        log(f"HK4 ERROR: {repr(e)}")
        set_status(f"HK4 ERROR: {type(e).__name__}")
    finally:
        _busy_lock.release()


def action_f5_steps_1to8():
    global _last_f5_ts
    now = time.time()
    if now - _last_f5_ts < MIN_INTERVAL_F5:
        return
    _last_f5_ts = now
    if not _busy_lock.acquire(blocking=False):
        return
    try:
        img = take_center_screenshot()
        prompt = (
            "You see a sequencing/ordering task with draggable text blocks.\n"
            "First, count how many blocks are visible (only 4, 5, 6, or 8).\nThen determine the correct order.\n\n"
            "IMPORTANT: output ONLY numbers 1..8 separated by spaces, no repeats, no other text.\n"
        )
        raw = _gemini_request_vision(img, prompt, api_key)
        log_block("F5 VISION RAW", raw)
        steps = parse_steps_1to8_unique(raw, max_n=8)
        if not steps:
            set_status("HK5: bad steps")
            return
        for s in steps:
            move_mouse_8_out_and_back(s)
        set_status(f"HK5 OK -> {' '.join(map(str, steps))}")
    except Exception as e:
        log(f"HK5 ERROR: {repr(e)}")
        set_status(f"HK5 ERROR: {type(e).__name__}")
    finally:
        _busy_lock.release()


def action_f6():
    global _last_f6_ts
    now = time.time()
    if now - _last_f6_ts < MIN_INTERVAL_F6:
        return
    _last_f6_ts = now
    if not _busy_lock.acquire(blocking=False):
        return
    try:
        img = take_center_screenshot()
        prompt = "Solve the multi-select question in the image.\nReturn ONLY the letters A..F separated by spaces.\nNo other text."
        raw = _gemini_request_vision(img, prompt, api_key)
        log_block("F6 VISION RAW", raw)
        answers = normalize_multi_answers_af(raw)
        if not answers:
            set_status("HK6: no A-F")
            return
        deg_map = {"A": 0, "B": 60, "C": 120, "D": 180, "E": 240, "F": 300}
        for ans in answers:
            move_mouse_deg_out_and_back(deg_map[ans], radius=HEX_R)
        set_status(f"HK6 OK -> {' '.join(answers)}")
    except Exception as e:
        log(f"HK6 ERROR: {repr(e)}")
        set_status(f"HK6 ERROR: {type(e).__name__}")
    finally:
        _busy_lock.release()


def toggle_window():
    global _window_visible
    if _window_visible:
        root.withdraw()
        _window_visible = False
    else:
        root.deiconify()
        root.lift()
        root.focus_force()
        _window_visible = True


def stop_everything_and_exit():
    global _hotkey_listener
    _stop_event.set()
    try:
        if _hotkey_listener is not None:
            _hotkey_listener.stop()
    except Exception:
        pass
    try:
        root.after(0, root.destroy)
    except Exception:
        pass


def _hotkeys_thread():
    global _hotkey_listener

    def wrap(fn):
        return lambda: threading.Thread(target=fn, daemon=True).start()

    combos = {
        "<ctrl>+<alt>+2": wrap(action_f2),
        "<ctrl>+<alt>+4": wrap(action_f4),
        "<ctrl>+<alt>+5": wrap(action_f5_steps_1to8),
        "<ctrl>+<alt>+6": wrap(action_f6),
        "<ctrl>+<alt>+h": lambda: root.after(0, toggle_window),
        "<ctrl>+<alt>+x": stop_everything_and_exit,
    }

    _hotkey_listener = keyboard.GlobalHotKeys(combos)
    _hotkey_listener.start()
    log("Hotkeys registered OK")
    set_status("Hotkeys: Ctrl+Opt+2/4/5/6, Ctrl+Opt+H show/hide, Ctrl+Opt+X exit")

    while not _stop_event.is_set():
        time.sleep(0.1)


def add_context_menu(entry: tk.Entry):
    menu = tk.Menu(entry, tearoff=0)
    menu.add_command(label="Paste", command=lambda: entry.event_generate("<<Paste>>"))
    menu.add_command(label="Copy", command=lambda: entry.event_generate("<<Copy>>"))
    menu.add_command(label="Cut", command=lambda: entry.event_generate("<<Cut>>"))

    def popup(e):
        menu.tk_popup(e.x_root, e.y_root)

    entry.bind("<Button-2>", popup)
    entry.bind("<Button-3>", popup)
    entry.bind("<Control-Button-1>", popup)


def open_log_file():
    _ensure_dir(LOG_FILE)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    subprocess.Popen(["open", "-e", LOG_FILE])


def start_bot():
    global api_key, _window_visible
    api_key = api_entry.get().strip()
    if not api_key:
        messagebox.showerror("Error", "API key required")
        return
    save_api(api_key)
    set_status("Starting hotkeys...")
    threading.Thread(target=_hotkeys_thread, daemon=True).start()
    _window_visible = False
    root.after(300, root.withdraw)


def on_close_app():
    stop_everything_and_exit()


# ================= GUI ====================
root = tk.Tk()
root.title("OCR Bot (Vision) — macOS")
root.geometry("580x500")
root.resizable(False, False)

def run_on_main(fn, timeout=3.0):
    """Выполнить fn() в главном потоке Tk и дождаться завершения."""
    done = threading.Event()
    box = {"err": None, "res": None}

    def job():
        try:
            box["res"] = fn()
        except Exception as e:
            box["err"] = e
        finally:
            done.set()

    root.after(0, job)
    done.wait(timeout=timeout)
    if box["err"]:
        raise box["err"]
    return box["res"]


if not require_valid_license_or_exit(root):
    root.destroy()
    sys.exit(1)

def label(parent, text, size=10, bold=False):
    return tk.Label(
        parent,
        text=text,
        fg="black",
        bg="SystemButtonFace",
        font=("Segoe UI", size, "bold" if bold else "normal"),
        anchor="w",
        justify="left",
    )

label(root, "🔑 Gemini API Key", 11, bold=True).pack(pady=(14, 6), padx=16, anchor="w")

api_entry = tk.Entry(root, width=78, show="*")
api_entry.pack(padx=16)
api_entry.insert(0, load_api())
api_entry.focus_set()
api_entry.select_range(0, tk.END)
add_context_menu(api_entry)

btns = tk.Frame(root)
btns.pack(pady=12)

ttk.Button(btns, text="Start", command=start_bot).pack(side="left", padx=6)
ttk.Button(btns, text="Show log", command=open_log_file).pack(side="left", padx=6)

label(
    root,
    "HOTKEYS:\n"
    "Ctrl+Option+2 = MC (A/B/C/D) -> move mouse\n"
    "Ctrl+Option+4 = solve task -> paste answer\n"
    "Ctrl+Option+5 = steps 1..8 -> smooth out-and-back\n"
    "Ctrl+Option+6 = multi-select A..F -> smooth out-and-back\n"
    "Ctrl+Option+H = show/hide\n"
    "Ctrl+Option+X = exit\n\n"
    "Permissions needed:\n"
    "- Privacy & Security → Accessibility (OCRBot)\n"
    "- Privacy & Security → Screen Recording (OCRBot)\n",
    9,
).pack(pady=12, padx=16, anchor="w")

status_var = tk.StringVar(value="Status: waiting for Start")
tk.Label(root, textvariable=status_var, fg="#0077cc", wraplength=560, justify="left", font=("Segoe UI", 9)).pack(
    pady=10, padx=16, anchor="w"
)

root.protocol("WM_DELETE_WINDOW", on_close_app)
root.mainloop()
