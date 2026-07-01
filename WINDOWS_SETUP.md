# Windows Setup Guide

This guide is for colleagues setting up the Autocallable Pricer on a native
Windows machine (no WSL/Linux required). For day-to-day usage once installed,
see [USER_MANUAL.md](USER_MANUAL.md).

All commands below are for **PowerShell** (the default terminal in Windows
Terminal / VS Code). CMD equivalents are noted where they differ.

---

## 1. Install prerequisites

### Git

1. Download from https://git-scm.com/download/win and run the installer.
2. Accept the defaults — they work fine for this project.
3. Verify in a new PowerShell window:
   ```powershell
   git --version
   ```

### Python 3.10+

1. Download from https://www.python.org/downloads/windows/ (use the
   "Windows installer (64-bit)" for the latest 3.x release).
2. **Important:** on the first installer screen, tick **"Add python.exe to PATH"**
   before clicking Install. This is unticked by default and is the #1 cause of
   `python` not being recognized afterwards.
3. Verify in a new PowerShell window:
   ```powershell
   python --version
   ```
   If this prints `Python was not found...` or opens the Microsoft Store,
   Python wasn't added to PATH — rerun the installer and repair/tick the box,
   or search "Manage App Execution Aliases" in Windows search and turn off
   the `python.exe` / `python3.exe` redirects there.

---

## 2. Clone the project

```powershell
cd C:\Users\<you>\Documents        # or wherever you keep projects
git clone <repo-url> autocall-pricer
cd autocall-pricer
```

---

## 3. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If activation fails with a message about "running scripts is disabled on this
system", PowerShell's execution policy is blocking it. Fix it once per user
account:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run `.venv\Scripts\Activate.ps1`. (Using CMD instead of PowerShell?
Activate with `.venv\Scripts\activate.bat` — no execution policy issue there.)

Your prompt should now start with `(.venv)`.

---

## 4. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs numpy, scipy, pandas, matplotlib, jupyter, and PyYAML.

---

## 5. Sanity check

```powershell
python main.py --no-fit-check
```

A successful run prints calibrated model parameters followed by three pricing
lines (Local Vol, Heston, SABR).

---

## 6. Everyday use

Same commands as the manual, just with `python` instead of `.venv/bin/python`
(activating the venv already puts the right `python` on PATH):

```powershell
python main.py
python scenarios/run_scenarios.py --n-paths 20000 --output results.csv
python scenarios/run_scenarios.py --greeks --n-paths 20000 --n-paths-greeks 10000
```

Forward slashes in paths (e.g. `scenarios/scenarios.yaml`) work fine on
Windows — Python accepts both `/` and `\`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python` not recognized | Reinstall Python with "Add to PATH" checked, or fix via "Manage App Execution Aliases" (see step 1). Restart the terminal after installing. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run the `Set-ExecutionPolicy` command in step 3, or use `activate.bat` in CMD instead. |
| `pip install` fails on a package needing a compiler | Rare for numpy/scipy on recent Python (they ship prebuilt wheels), but if it happens, install the latest Python 3.11/3.12 rather than a very new pre-release version — wheel availability lags new Python releases. |
| Antivirus/Defender flags the venv or blocks script execution | Whitelist the project folder, or exclude `.venv\` in Windows Security > Virus & threat protection > Exclusions. |
| Terminal doesn't show `(.venv)` after activation | You're in a different shell than you activated in (e.g. opened a new tab). Re-run the activate command in that shell. |
