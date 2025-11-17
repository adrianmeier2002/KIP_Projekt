# RL Training & Visualisierung - Anleitung

## Übersicht

Diese Anleitung erklärt, wie Sie den DQN-Agenten trainieren und die Visualisierung verwenden.

## Training starten

### Über die GUI (Empfohlen)

1. **Anwendung starten:**
   ```bash
   python main.py
   ```

2. **Training starten:**
   - Klicken Sie auf: **Menü → Reinforcement Learning → Training starten (mit Visualisierung)**
   - Im ersten Dialog: Anzahl Episoden eingeben (z.B. `1000`)
   - Im zweiten Dialog: Schwierigkeit wählen:
     - **Leicht**: Weniger Minen, einfacher zum Lernen
     - **Mittel**: Standard-Schwierigkeit
     - **Schwer**: Mehr Minen, schwieriger

3. **Training läuft:**
   - Das Training läuft im Hintergrund
   - Alle **100 Episoden** wird automatisch eine Visualisierung gestartet
   - Am **Ende** folgt ein finaler Test-Lauf

### Über die Kommandozeile (Ohne GUI)

```bash
python -m src.reinforcement_learning.trainer ^
  --episodes 1500 ^
  --difficulty easy ^
  --width 9 --height 9 ^
  --eval-episodes 20
```

**Wichtige Parameter:**
- `--episodes`: Anzahl Episoden (Standard: 1000)
- `--difficulty`: easy, medium, hard (Standard: medium)
- `--width / --height`: Override für Brettgröße (Standard: Werte aus `src/utils/constants.py`)
- `--save-path`: Pfad zum Speichern (Standard: models/dqn_model.pth)
- `--log-interval`: Episoden zwischen Logging (Standard: 100)
- `--use-flags`: Flag- UND Reveal-Aktionen trainieren (Standard deaktiviert → Reveal-only wie im Referenzprojekt von [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning))
- `--eval-episodes`: Anzahl greedy Evaluations-Episoden pro Log-Block (ε=0). Setze `0`, falls nur Trainingsstatistiken benötigt werden.

Nach jedem Log-Block werden nun zwei Werte ausgegeben:
1. **Trainings-Winrate** (inkl. Exploration, d.h. mit aktuellem ε)
2. **Greedy Evaluation** (ε=0) → echte Gewinnrate ohne Zufallszüge, plus durchschnittliche Zuganzahl.

## Visualisierung verstehen

### Automatische Visualisierungen (alle 100 Episoden)

Während des Trainings:
1. Nach jeder 100. Episode erscheint automatisch eine Visualisierung
2. Der Agent spielt eine komplette Episode
3. Sie sehen in Echtzeit, wie der Agent Zellen aufdeckt
4. Nach der Episode wird das Ergebnis angezeigt (Gewonnen/Verloren)

### Finaler Test-Lauf

Am Ende des Trainings:
- Ein finaler Test-Lauf wird automatisch gestartet
- Schnellere Visualisierung (50ms Delay statt 100ms)
- Zeigt die finale Leistung des trainierten Agents

## Geladenes Modell testen

### Modell laden

1. **Menü → Reinforcement Learning → Modell laden und testen**
2. Wählen Sie die Modell-Datei (z.B. `models/dqn_model.pth`)
3. Klicken Sie OK

### Agent manuell spielen lassen

1. Im **RL-Visualizer-Widget** (unterhalb des Spielfelds) finden Sie:
   - Button: **"RL-Agent spielen lassen"**
   - Button: **"Stoppen"**

2. **Agent starten:**
   - Klicken Sie auf "RL-Agent spielen lassen"
   - Der Agent spielt automatisch eine Episode
   - Sie sehen in Echtzeit die Entscheidungen

3. **Agent stoppen:**
   - Klicken Sie auf "Stoppen" um die aktuelle Episode zu beenden

## Tipps für effektives Training

### Training-Parameter

- **Episoden**: Starten Sie mit 1000 Episoden für erste Tests
- **Schwierigkeit**: Beginnen Sie mit "Leicht" für schnelleres Lernen
- **Wartezeit**: Die Visualisierung kann mit dem Delay-Parameter angepasst werden

### Was zu beachten ist

1. **Training dauert**: Je nach Episodenanzahl kann Training mehrere Minuten/Stunden dauern
2. **GUI bleibt aktiv**: Sie können während des Trainings weiterhin manuell spielen
3. **Modell wird gespeichert**: Das Modell wird automatisch gespeichert (Standard: `models/dqn_model.pth`)

## Troubleshooting

### Training startet nicht

- Überprüfen Sie, ob PyTorch korrekt installiert ist
- Prüfen Sie die Konsole auf Fehlermeldungen

### Visualisierung erscheint nicht

- Warten Sie bis Episode 100, 200, 300, etc. erreicht wird
- Prüfen Sie, ob das Training läuft (Konsole)

### Modell kann nicht geladen werden

- Stellen Sie sicher, dass das Modell bereits trainiert wurde
- Überprüfen Sie den Pfad zur Modell-Datei
- Prüfen Sie, ob die Modell-Datei existiert (`models/dqn_model.pth`)

## Fortgeschrittene Nutzung

### Training fortsetzen

Sie können ein bereits trainiertes Modell laden und weiter trainieren (zukünftige Erweiterung).

### Verschiedene Schwierigkeitsgrade

- Trainieren Sie verschiedene Modelle für verschiedene Schwierigkeitsgrade
- Speichern Sie sie mit unterschiedlichen Namen (z.B. `dqn_model_easy.pth`, `dqn_model_hard.pth`)

## Beispiel-Workflow

1. **Start**: `python main.py`
2. **Training starten**: Menü → RL → Training starten
   - 1000 Episoden, Mittel
3. **Warten**: Beobachten Sie die Visualisierungen alle 100 Episoden
4. **Modell laden**: Nach Training → Menü → RL → Modell laden
5. **Testen**: "RL-Agent spielen lassen" klicken
6. **Beobachten**: Sehen Sie, wie gut der Agent spielt!

