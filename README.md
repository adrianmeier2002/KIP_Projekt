# KIP_Projekt - Minesweeper mit Hybrid AI (Constraint Solver + Reinforcement Learning)

Ein vollständiges Minesweeper-Spiel mit GUI (PySide6) und **Hybrid AI**: Constraint Solver kombiniert mit Deep Q-Network (DQN).

## 🎯 Projektübersicht

Dieses Projekt implementiert:
- **Minesweeper-Spiel** mit konfigurierbaren Spielfeldern
- **GUI** mit PySide6 für manuelles Spielen
- **Hybrid AI** = **Constraint Solver (100% sichere Züge) + Reinforcement Learning (für Guess-Situationen)**
- **3 Schwierigkeitsgrade**: Leicht (~10%), Mittel (~15%), Schwer (~20%)

## 🚀 Warum Hybrid AI?

### **Das Problem mit reinem Reinforcement Learning:**

Unsere Tests zeigten:
- **Reines DQN:** 0% Win-Rate nach 5000 Episoden auf 7x7 Board
- **Problem:** Minesweeper erfordert logische Inferenz, die CNNs nicht gut lernen können
- **Lösung:** **Hybrid-Ansatz!**

### **Hybrid AI = Best of Both Worlds:**

```
1. Constraint Solver findet 100% sichere Züge
   → Nutzt Minesweeper-Logik (z.B. "Zelle zeigt 1, 1 Flagge → Rest ist sicher")
   
2. RL nur für Guess-Situationen
   → Wenn keine sicheren Züge verfügbar, wählt RL den "besten" Guess
   
3. Ergebnis:
   ✅ Agent macht keine vermeidbaren Fehler
   ✅ RL lernt nur für schwierige Guess-Situationen
   ✅ Deutlich höhere Win-Rate!
```


## 🛠 Installation

### Voraussetzungen

- **Python 3.8 oder höher** muss installiert sein
- **Empfohlen:** Python 3.10 oder neuer für beste Kompatibilität
- **Optional:** CUDA-fähige GPU für schnelleres Training (CPU funktioniert auch)

### Schritt-für-Schritt Installation

#### 1. Repository klonen oder herunterladen

```bash
# Mit Git:
git clone https://github.com/your-repo/KIP_Projekt.git
cd KIP_Projekt

# Oder: ZIP-Datei herunterladen und entpacken
```

#### 2. Virtuelle Umgebung erstellen (empfohlen)

```bash
# Virtuelle Umgebung erstellen
python -m venv venv

# Aktivieren:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### 3. Abhängigkeiten installieren

```bash
pip install -r requirements.txt

# Für Windows (falls pip nicht funktioniert):
python -m pip install -r requirements.txt
```

#### 4. Installation überprüfen

```bash
# Tests ausführen um sicherzustellen, dass alles funktioniert:
pytest tests/ -v
```

**Alle 81 Tests sollten bestehen!** ✅

### Mögliche Probleme und Lösungen

**Problem:** PyTorch-Fehler unter Windows  
**Lösung:** Installieren Sie [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

**Problem:** Import-Fehler bei PySide6  
**Lösung:** Neuinstallation: `pip install --upgrade --force-reinstall PySide6`

**Problem:** Tests schlagen fehl  
**Lösung:** Stellen Sie sicher, dass Sie im Projektverzeichnis sind und alle Dependencies installiert sind

---

## 🎮 Verwendung

### **Option 1: GUI starten (Empfohlen für Einsteiger)**

```bash
python main.py
```

Das Spiel öffnet sich in einem Fenster. Sie können:

**Spielen:**
- **Linksklick:** Zelle aufdecken
- **Rechtsklick:** Flagge setzen/entfernen
- **Power-Ups nutzen:** Radar (70P), Scanner (70P), Blitz (50P)
- **Neues Spiel:** Menü → Spiel → Neues Spiel (Leicht/Mittel/Schwer)
- **Spielfeldgröße ändern:** Menü → Spiel → Spielfeldgröße ändern

**RL-Training:**
- **Training starten:** Menü → Reinforcement Learning → Training starten
- **Modell laden:** Menü → Reinforcement Learning → Modell laden und testen
- Der Agent wird dann das Spiel automatisch spielen und Sie können ihm zusehen!

### **Option 2: Kommandozeile Training (Hybrid Mode - EMPFOHLEN)**

```bash
python -m src.reinforcement_learning.trainer \
  --episodes 1000 \
  --difficulty medium \
  --width 7 --height 7 \
  --save-path models/hybrid_7x7.pth \
  --eval-episodes 25
```

**Wichtig:** Hybrid-Modus ist **standardmäßig aktiviert**!

**Parameter:**
- `--episodes`: Anzahl Trainingsepisoden (Standard: 1000)
- `--difficulty`: easy, medium, hard (Standard: medium)
- `--width / --height`: Brettgröße (Optional)
- `--save-path`: Modell-Speicherpfad
- `--eval-episodes`: Evaluations-Episoden pro Log (zeigt echte Win-Rate)
- `--no-hybrid`: **Deaktiviert Hybrid-Modus** (nicht empfohlen!)

### **Option 3: Pure RL (Zum Vergleich - NICHT EMPFOHLEN)**

```bash
python -m src.reinforcement_learning.trainer \
  --episodes 5000 \
  --difficulty easy \
  --width 5 --height 5 \
  --no-hybrid \
  --save-path models/pure_rl_5x5.pth
```

**Erwartung:** Win-Rate bleibt bei ~0%. Nur für Benchmarking!

---

## 📈 Training verstehen

### **Wichtige Metriken:**

```
Episode 100/1000
  Avg Reward: 45.23
  Avg Length: 12.4          ← Züge pro Episode (höher = besser)
  Win Rate: 15.0%           ← Training Win-Rate (mit Exploration)
  Epsilon: 0.825
  Eval (ε=0) → Win Rate: 35.0% | Avg Len: 15.2    ← ECHTE Win-Rate!
  Solver Usage → 65.3% | RL Guesses: 34.7%        ← Hybrid-Statistik
```

**Was die Zahlen bedeuten:**

- **Avg Length:** Wie lange überlebt der Agent?
  - Zu niedrig (<5 auf 7x7) = stirbt zu früh
  - Gut (>10 auf 7x7) = macht Fortschritt

- **Win Rate (Training):** Mit Exploration (epsilon)
  - Oft niedriger wegen Zufallszügen

- **Eval Win Rate (ε=0):** **WICHTIGSTE METRIK!**
  - Ohne Exploration = echte Policy-Qualität
  - Sollte kontinuierlich steigen

- **Solver Usage:** % der Züge vom Constraint Solver
  - 60-80% = gut (viele sichere Züge genutzt)
  - <30% = Problem (wenige sichere Situationen)

### **Lern-Verlauf (typisch):**

```
Episodes 1-200:    Win-Rate: 10-20% (Exploration)
Episodes 200-500:  Win-Rate: 20-40% (Lernt Guess-Strategie)
Episodes 500-1000: Win-Rate: 30-60% (Konvergenz)
```

**Wenn Win-Rate bei 0% stagniert:**
- Überprüfen Sie: Ist Hybrid-Modus aktiv? (`--no-hybrid` NICHT verwenden)
- Reduzieren Sie Schwierigkeit (easy statt medium)
- Reduzieren Sie Brettgröße (5x5 statt 7x7)

---

## 🏗 Architektur

### **Komponenten:**

```
src/
├── minesweeper/              # Spiel-Logik
│   ├── board.py              # Spielfeld-Verwaltung
│   ├── cell.py               # Zellen-Logik
│   └── game.py               # Spiel-Steuerung
│
├── gui/                      # GUI (PySide6)
│   ├── game_board.py
│   ├── main_window.py
│   └── rl_visualizer.py
│
└── reinforcement_learning/   # AI
    ├── constraint_solver.py  # ✨ Findet 100% sichere Züge
    ├── hybrid_agent.py       # ✨ Kombiniert Solver + RL
    ├── dqn_agent.py          # Deep Q-Network (RL-Teil)
    ├── network.py            # CNN-Architektur
    ├── environment.py        # RL-Environment
    └── trainer.py            # Training-Loop
```

### **Hybrid Agent - Funktionsweise:**

```python
def select_action(state, game):
    # 1. Versuche Constraint Solver
    safe_moves = solver.get_safe_moves(game)
    if safe_moves:
        return random.choice(safe_moves)  # 100% sicher!
    
    # 2. Kein sicherer Zug → RL wählt "besten" Guess
    return dqn_agent.select_action(state)
```

### **State Representation (9 Kanäle):**

```
Channel 0: Basis-Encoding (-0.8 hidden, -0.5 flag, -1 mine, 0-1 number)
Channel 1: Hidden-Maske
Channel 2: Flag-Maske
Channel 3: Aufgedeckte Zahlen
Channel 4: Verdeckte Nachbarn
Channel 5: Geflaggte Nachbarn
Channel 6: Hinweis-Summe
Channel 7: Frontier-Maske (neben aufgedeckten Zellen)
Channel 8: Safe-Cell-Maske (100% sichere Zellen)
```

### **Network Architecture:**

```
Input: (9, H, W)
Conv Stack (4 × 128 filters) + BatchNorm + ReLU
AdaptiveAvgPool2d(8×8)
FC: 8192 → 512 → 512 → num_actions
```

---

## 🧪 Tests

Das Projekt enthält umfassende Tests (81 Tests insgesamt):

```bash
# Alle Tests ausführen (empfohlen)
pytest tests/ -v

# Oder mit Python-Modul:
python -m pytest tests/ -v

# Nur RL-Tests
pytest tests/reinforcement_learning/ -v

# Nur Minesweeper-Tests  
pytest tests/minesweeper/ -v

# Mit Coverage-Report
pytest tests/ --cov=src --cov-report=html
```

**Alle 81 Tests sollten bestehen!** Wenn nicht, überprüfen Sie Ihre Installation.

---

## 📚 Dokumentation

Zusätzliche Dokumentation finden Sie in folgenden Dateien:

- **[RL_IMPLEMENTATION_GUIDE.md](docs/RL_IMPLEMENTATION_GUIDE.md)**: Technische Details zur RL-Implementierung
- **[RL_TRAINING_GUIDE.md](docs/RL_TRAINING_GUIDE.md)**: Detaillierte Trainingsanleitung
- **[CHANGELOG_RL_FIX_V3.md](CHANGELOG_RL_FIX_V3.md)**: Versionshistorie und Änderungen

### Projekt-Struktur

```
KIP_Projekt/
├── main.py                          # Haupteinstiegspunkt (GUI starten)
├── requirements.txt                 # Python-Abhängigkeiten
├── README.md                        # Diese Datei
├── src/
│   ├── minesweeper/                # Spiellogik
│   │   ├── board.py                # Spielfeld-Verwaltung
│   │   ├── cell.py                 # Zellen-Logik
│   │   └── game.py                 # Spiel-Steuerung (inkl. Power-Ups)
│   ├── gui/                        # Grafische Benutzeroberfläche
│   │   ├── main_window.py          # Hauptfenster
│   │   ├── game_board.py           # Spielfeld-Widget
│   │   ├── menu_bar.py             # Menüleiste
│   │   └── rl_visualizer.py        # RL-Agent Visualisierung
│   ├── reinforcement_learning/     # KI-Komponenten
│   │   ├── constraint_solver.py    # ✨ Regelbasierter Solver (100% sichere Züge)
│   │   ├── hybrid_agent.py         # ✨ Hybrid-Agent (Solver + RL)
│   │   ├── dqn_agent.py            # Deep Q-Network Agent
│   │   ├── network.py              # Neuronales Netzwerk
│   │   ├── environment.py          # RL-Environment Wrapper
│   │   └── trainer.py              # Training-Skript
│   └── utils/
│       └── constants.py            # Konstanten und Konfiguration
├── tests/                          # Unit-Tests (81 Tests)
│   ├── minesweeper/                # Tests für Spiellogik
│   └── reinforcement_learning/    # Tests für RL-Komponenten
├── models/                         # Trainierte Modelle (wird erstellt)
└── docs/                           # Zusätzliche Dokumentation
```

---

## 🎓 Lessons Learned

### **Warum Reines RL scheiterte:**

1. **Sparse Rewards:** Agent stirbt in 95% der Fälle sofort
2. **Kombinatorische Explosion:** Zu viele Zustände
3. **Logische Inferenz:** CNNs können Minesweeper-Regeln nicht effektiv lernen
4. **Sample-Ineffizienz:** Braucht Millionen statt Tausende Episoden

### **Warum Hybrid funktioniert:**

1. **Sichere Züge garantiert:** Solver macht keine Fehler
2. **RL nur für Guesses:** Fokussiert auf schwierige Situationen
3. **Domänenwissen:** Minesweeper-Logik integriert
4. **Sample-Effizienz:** Lernt schneller durch weniger Fehler

### **Generelle Erkenntnis:**

**Für Constraint-basierte Probleme:**
- Hybrid-Ansätze (Rule-Based + ML) > Reines ML
- Domänenwissen beschleunigt Lernen massiv
- CNNs sind schlecht in kombinatorischer Logik


## 🛠 Technologie-Stack

- **Python 3.8+**: Programmiersprache
- **PySide6 6.10+**: GUI Framework (Qt für Python)
- **PyTorch 2.9+**: Deep Learning Framework
- **NumPy 2.3+**: Numerische Berechnungen
- **Pytest 9.0+**: Testing Framework

## 🎮 Spiel-Features und Herausforderungen

Dieses Minesweeper-Spiel bietet zusätzliche Features und Herausforderungen, die das klassische Spiel erweitern:

### **Power-Ups (mit Punktesystem)**

Durch das Aufdecken von Feldern verdienen Sie Punkte (1 Punkt pro Feld). Diese Punkte können Sie für Power-Ups ausgeben:

#### **📡 Radar (70 Punkte)**
- **Funktion:** Deckt einen 3×3-Bereich sofort auf
- **Besonderheit:** Minen in diesem Bereich werden **nicht** ausgelöst, sondern nur mit einem Warnsymbol ⚠️ markiert
- **Verwendung:** Klicken Sie auf den Radar-Button und dann auf die gewünschte Zelle als Zentrum
- **Tipp:** Ideal für gefährliche Bereiche, wo Sie vermuten, dass Minen sein könnten

#### **🔍 Scanner (70 Punkte)**
- **Funktion:** Zählt die Anzahl der Minen in einem 3×3-Bereich
- **Anzeige:** Das Ergebnis (z.B. "🔍3") wird auf der gescannten Zelle angezeigt
- **Verwendung:** Klicken Sie auf den Scanner-Button und dann auf die gewünschte Zelle als Zentrum
- **Tipp:** Nutzen Sie dies, um zusätzliche Informationen für Ihre Strategie zu erhalten

#### **⚡ Blitz (50 Punkte)**
- **Funktion:** Deckt automatisch 1-3 sichere Felder auf
- **Intelligenz:** Nutzt den Constraint-Solver, um nur **garantiert sichere** Felder aufzudecken
- **Verwendung:** Einfach auf den Blitz-Button klicken
- **Tipp:** Perfekt, wenn Sie feststecken und keine sichere Wahl sehen

### **Herausforderungen**

Das Spiel fügt dynamisch Herausforderungen hinzu, um das Spielerlebnis spannender zu gestalten:

#### **❓ Mystery-Felder**
- **Was ist das?** Aufgedeckte Zahlenfelder werden zu Fragezeichen ❓
- **Problem:** Sie können die echte Zahl nicht sehen!
- **Lösung:** Zahlen Sie 20 Punkte, um die Mystery-Zahl zu enthüllen
- **Häufigkeit:** Erscheint alle 15-25 aufgedeckten Felder
- **Strategie:** Überlegen Sie gut, ob Sie die Punkte ausgeben möchten oder ob Sie auch ohne diese Information weiterkommen

#### **⚡ Speed-Felder**
- **Was ist das?** Ein aufgedecktes Feld startet einen 5-Sekunden-Timer
- **Anzeige:** Großer roter Timer oben im Fenster mit Countdown
- **Ziel:** Decken Sie ein weiteres Feld auf, bevor die Zeit abläuft!
- **Konsequenz:** Wenn die Zeit abläuft → **Game Over**
- **Häufigkeit:** Erscheint alle 20-30 aufgedeckten Felder
- **Strategie:** Halten Sie immer sichere Züge bereit für den Fall, dass ein Speed-Feld erscheint

#### **🎮 Tetris-Felder**
- **Was ist das?** Ein aufgedecktes Feld aktiviert den Tetris-Modus
- **Anzeige:** Oben erscheint eine Tetris-Form (I, O, T, S, Z, L oder J)
- **Ziel:** Platzieren Sie diese Form auf dem Spielfeld
- **Regel:** Die Form muss vollständig auf verdeckte, minenfreie Felder passen
- **Vorschau:** Beim Hover über dem Feld sehen Sie, wo die Form platziert würde
- **Effekt:** Alle Felder der Form werden gleichzeitig aufgedeckt
- **Häufigkeit:** Erscheint alle 25-40 aufgedeckten Felder
- **Strategie:** Versuchen Sie, die Form in einem sicheren Bereich zu platzieren

### **Herausforderungen deaktivieren**

Für das RL-Training können Herausforderungen optional deaktiviert werden:
- Im Code: `Game(..., enable_challenges=False)`
- Dies ermöglicht fokussiertes Training ohne zufällige Komplikationen

---

## 🚀 Quick Start für Ungeduldige

```bash
# 1. Installation
pip install -r requirements.txt

# 2. Spiel starten
python main.py

# 3. Optional: Training starten (Hybrid Mode, 7x7, Medium)
python -m src.reinforcement_learning.trainer \
  --episodes 1500 \
  --difficulty medium \
  --width 7 --height 7 \
  --save-path models/hybrid_7x7.pth
```

**Viel Erfolg! 🎉**

