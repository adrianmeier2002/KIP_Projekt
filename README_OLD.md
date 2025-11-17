# KIP_Projekt - Minesweeper mit Reinforcement Learning

Ein vollständiges Minesweeper-Spiel mit GUI (PySide6) und Reinforcement Learning (DQN mit PyTorch).

## Projektübersicht

Dieses Projekt implementiert:
- **Minesweeper-Spiel** mit 20x30 Spielfeld
- **GUI** mit PySide6 für manuelles Spielen
- **Reinforcement Learning** mit Deep Q-Network (DQN) basierend auf PyTorch
- **3 Schwierigkeitsgrade**: Leicht (~10%), Mittel (~15%), Schwer (~20%)

## Projektstruktur

```
KIP_Projekt/
├── src/
│   ├── minesweeper/          # Spiellogik
│   ├── gui/                  # GUI-Komponenten
│   ├── reinforcement_learning/ # RL-Implementierung
│   └── utils/                # Hilfsfunktionen
├── models/                   # Gespeicherte RL-Modelle
├── data/                     # Training-Daten/Logs
├── requirements.txt
└── main.py                   # Hauptprogramm
```

## Installation

### Voraussetzungen

Python 3.8 oder höher muss installiert sein. Falls Python noch nicht installiert ist:

**Python von python.org**
1. Besuchen Sie https://www.python.org/downloads/
2. Laden Sie die neueste Python-Version herunter (3.8+)
3. Während der Installation: **Wichtig:** Aktivieren Sie "Add Python to PATH"
4. Installation abschließen

**Installation überprüfen:**
```powershell
python --version
# oder
py --version
```

### Projekt-Setup

1. Repository klonen oder herunterladen
2. Abhängigkeiten installieren:
```bash
# Wenn Python im PATH ist:
pip install -r requirements.txt

# Falls pip nicht gefunden wird, versuchen Sie:
python -m pip install -r requirements.txt
# oder
py -m pip install -r requirements.txt
```

**Hinweis für Windows:** Falls `pip` nicht erkannt wird, verwenden Sie `python -m pip` oder `py -m pip`.

### PyTorch Installation (Windows)

PyTorch benötigt die **Visual C++ Redistributable**. Falls beim Import von PyTorch ein Fehler auftritt:

**Visual C++ Redistributable installieren**
1. Laden Sie die Visual C++ Redistributable herunter:
   - https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Installieren Sie die Datei
3. PowerShell/Terminal neu starten

## Verwendung

### Manuelles Spielen

Starten Sie das GUI mit:
```bash
python main.py
```

**Bedienung:**
- Linksklick: Zelle aufdecken
- Rechtsklick: Flagge setzen/entfernen
- Menü: Neues Spiel mit verschiedenen Schwierigkeitsgraden starten
- Sichere Zellen werden automatisch aufgedeckt

### Reinforcement Learning Training

#### Training mit Visualisierung (GUI)

1. **Anwendung starten:**
   ```bash
   python main.py
   ```

2. **Training starten:**
   - Menü → **Reinforcement Learning** → **Training starten (mit Visualisierung)**
   - Anzahl Episoden eingeben (z.B. 1000)
   - Schwierigkeit wählen (Leicht/Mittel/Schwer)

3. **Visualisierung:**
   - Alle **100 Episoden** wird automatisch eine Visualisierung gestartet
   - Am **Ende** folgt ein finaler Test-Lauf
   - Sie sehen in Echtzeit, wie der Agent spielt!

#### Training über Kommandozeile (ohne GUI)

```bash
python -m src.reinforcement_learning.trainer ^
  --episodes 1200 ^
  --difficulty medium ^
  --width 15 --height 10 ^
  --eval-episodes 25 ^
  --save-path models/dqn_model_15x10.pth
```

**Parameter (Auszug):**
- `--episodes`: Anzahl der Trainingsepisoden (Standard: 1000)
- `--difficulty`: Schwierigkeitsgrad - easy, medium, hard (Standard: medium)
- `--width / --height`: Optionales Spielfeld-Override (Standard: Werte aus `constants.py`)
- `--save-path`: Pfad zum Speichern des Modells (Standard: models/dqn_model.pth)
- `--log-interval`: Episoden zwischen Logging (Standard: 100)
- `--use-flags`: Flag- und Aufdeck-Aktionen trainieren (Standard: ausgeschaltet → Reveal-only wie im Referenzprojekt)
- `--eval-episodes`: Anzahl greedy Evaluations-Episoden pro Log-Block (zeigt echte Gewinnrate ohne Exploration)

Jeder Log-Eintrag enthält nun zusätzlich eine **greedy Evaluation (ε=0)**, sodass sofort sichtbar ist, wie gut das Modell ohne Zufallszüge abschneidet. Am Ende des Trainings wird automatisch eine größere Evaluationsserie (mind. 25 Episoden) ausgeführt und zusammengefasst.

### Geladenes Modell testen

1. **Modell laden:**
   - Menü → **Reinforcement Learning** → **Modell laden und testen**
   - Wählen Sie die Modell-Datei (`models/dqn_model.pth`)

2. **Agent spielen lassen:**
   - Klicken Sie auf **"RL-Agent spielen lassen"** im RL-Visualizer-Widget
   - Der Agent spielt automatisch eine Episode

**Detaillierte Anleitung:** Siehe [docs/RL_TRAINING_GUIDE.md](docs/RL_TRAINING_GUIDE.md)

## Technische Details

- **Programmiersprache**: Python 3.8+
- **GUI Framework**: PySide6
- **RL Framework**: PyTorch
- **Algorithmus**: Deep Q-Network (DQN) mit Experience Replay und Target Network

## Features

- ✅ Vollständiges Minesweeper-Spiel mit 20x30 Spielfeld
- ✅ 3 Schwierigkeitsgrade (Leicht, Mittel, Schwer)
- ✅ Manuelles Spielen mit intuitiver GUI
- ✅ Automatisches Aufdecken sicherer Zellen
- ✅ DQN-basiertes Reinforcement Learning
- ✅ Training-Script mit Logging
- ✅ Modell-Speicherung und -Laden
- ✅ **RL-Visualisierung während des Trainings (alle 100 Episoden)**
- ✅ **Finaler Test-Lauf am Ende des Trainings**
- ✅ **Manuelles Testen geladener Modelle in der GUI**
- ✅ **Reveal-only Action Space (Flags optional) – inspiriert durch [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning) für stabileres Training**
- ✅ **Lineare Epsilon-Schedules & automatische greedy Evaluationsläufe (Win-Rate ohne Zufall)**
- ✅ **CLI-Training mit frei wählbarer Brettgröße (`--width/--height`) und Flag-Modus**

## Entwicklung

Das Projekt wurde mit Hilfe von Cursor und KI-Assistenz erstellt, um zu testen, wie effektiv KI bei der Programmierung helfen kann.
