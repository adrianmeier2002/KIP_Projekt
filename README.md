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

## 📊 Erwartete Performance

### **Mit Hybrid Agent:**

| Board Size | Difficulty | Expected Win-Rate | Training Time |
|------------|------------|-------------------|---------------|
| 5x5 | Easy | 40-70% | 500-1000 Episodes |
| 7x7 | Medium | 20-50% | 1000-2000 Episodes |
| 9x9 | Medium | 15-35% | 2000-3000 Episodes |

**Wichtig:** Der Solver allein kann bereits ~30-60% lösen (je nach Schwierigkeit)!

### **Ohne Hybrid (Pure RL):**

| Board Size | Difficulty | Win-Rate | Problem |
|------------|------------|----------|---------|
| 5x5+ | Any | ~0% | Lernt nicht effektiv |

**→ Hybr

id-Modus ist STANDARD und EMPFOHLEN!**

---

## 🛠 Installation

### Voraussetzungen

Python 3.8+ muss installiert sein.

### Projekt-Setup

```bash
# 1. Repository klonen
git clone https://github.com/your-repo/KIP_Projekt.git
cd KIP_Projekt

# 2. Abhängigkeiten installieren
pip install -r requirements.txt

# Für Windows:
python -m pip install -r requirements.txt
```

**Windows:** Falls PyTorch-Fehler auftreten, installieren Sie [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

---

## 🎮 Verwendung

### **Option 1: GUI (Manuelles Spielen + Training)**

```bash
python main.py
```

**Features:**
- Linksklick: Zelle aufdecken
- Rechtsklick: Flagge setzen/entfernen
- Menü → RL Training starten (mit Visualisierung)
- Menü → Modell laden und testen

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

```bash
# Alle Tests
python -m pytest tests/ -v

# Nur RL-Tests
python -m pytest tests/reinforcement_learning/ -v

# Nur Minesweeper-Tests
python -m pytest tests/minesweeper/ -v
```

---

## 📚 Dokumentation

- **[RL_IMPLEMENTATION_GUIDE.md](docs/RL_IMPLEMENTATION_GUIDE.md)**: Technische Details
- **[RL_TRAINING_GUIDE.md](docs/RL_TRAINING_GUIDE.md)**: Training-Anleitung
- **[CHANGELOG_RL_FIX_V3.md](CHANGELOG_RL_FIX_V3.md)**: Versionshistorie

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

---

## 🔬 Weiterführende Experimente

### **Experiment 1: Solver vs. RL Performance**

```bash
# Pure Solver (keine RL)
# → Baseline Win-Rate messen

# Pure RL (--no-hybrid)
# → Zeigt RL-Limitation

# Hybrid
# → Beste Performance
```

### **Experiment 2: Schwierigkeitsgrade**

```bash
# Easy: 10% Minen → 50-70% Win-Rate erwartet
# Medium: 15% Minen → 30-50% Win-Rate erwartet  
# Hard: 20% Minen → 15-35% Win-Rate erwartet
```

### **Experiment 3: Brettgrößen**

```bash
# 5x5: Schnelles Training, höhere Win-Rate
# 7x7: Moderates Training, mittlere Win-Rate
# 9x9+: Langsames Training, niedrige Win-Rate
```

---

## 🤝 Mitwirkende

Projekt entwickelt mit KI-Assistenz (Cursor + Claude Sonnet) zur Evaluation von AI-gestützter Programmierung.

---

## 📄 Lizenz

[Lizenz hier einfügen]

---

## 🚀 Quick Start für Ungeduldige

```bash
# Installation
pip install -r requirements.txt

# Training starten (Hybrid Mode, 7x7, Medium)
python -m src.reinforcement_learning.trainer \
  --episodes 1500 \
  --difficulty medium \
  --width 7 --height 7 \
  --save-path models/hybrid_7x7.pth

# Erwartung: Win-Rate steigt auf 25-45%!
```

**Viel Erfolg! 🎉**

