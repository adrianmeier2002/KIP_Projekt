# Implementierungs-Guide: Minesweeper mit RL

Dieses Dokument erklärt die Implementierung jedes Teils des Projekts und die Design-Entscheidungen.

## Phase 1: Projekt-Setup und Grundstruktur

### Was wurde gemacht:

1. **Projektstruktur erstellt** - Modulare Struktur:
   - `src/minesweeper/` - Spiellogik (Kern)
   - `src/gui/` - GUI-Komponenten
   - `src/reinforcement_learning/` - RL-Implementierung
   - `src/utils/` - Hilfsfunktionen

2. **Konstanten definiert** (`src/utils/constants.py`):
   - Spielfeldgröße: 20x30 (600 Zellen)
   - 3 Schwierigkeitsgrade:
     - Leicht: ~10% Minen (~60 Minen)
     - Mittel: ~15% Minen (~90 Minen) - Standard
     - Schwer: ~20% Minen (~120 Minen)
   - Zellzustände: HIDDEN, REVEALED, FLAGGED, MINE

3. **Dependencies** (`requirements.txt`):
   - PySide6: GUI Framework
   - PyTorch: Deep Learning für RL
   - NumPy: Numerische Operationen

---

## Phase 2: Minesweeper Kern-Logik

### 2.1 Cell-Klasse (`src/minesweeper/cell.py`)

**Zweck:** Repräsentiert eine einzelne Zelle im Spielfeld.

**Implementierung:**
- **Zustandsverwaltung:** Jede Zelle hat einen Zustand (verdeckt, aufgedeckt, markiert)
- **Minen-Markierung:** `set_mine()` markiert die Zelle als Mine
- **Nachbarzählung:** `adjacent_mines` speichert die Anzahl benachbarter Minen
- **Reveal-Logik:** `reveal()` deckt die Zelle auf (nur wenn verdeckt)
- **Flag-Logik:** `flag()` togglet die Flagge (nur wenn verdeckt)

**Design-Entscheidungen:**
- Verwendet Konstanten statt Magic Numbers für bessere Lesbarkeit
- Rückgabewerte (`True`/`False`) zeigen an, ob Operation erfolgreich war
- Getter-Methoden (`is_revealed()`, `is_flagged()`, `is_hidden()`) für klare API

**Mögliche Optimierungen:**
- ✅ Aktuell: Gut strukturiert, keine Optimierungen nötig

---

### 2.2 Board-Klasse (`src/minesweeper/board.py`)

**Zweck:** Verwaltet das gesamte Spielfeld und die Minen-Platzierung.

**Implementierung:**
- **Spielfeld-Generation:** Erstellt 2D-Array von Cell-Objekten
- **Minen-Platzierung:** `place_mines()` platziert Minen zufällig (ausschließlich erster Klick)
- **Nachbarzählung:** `_calculate_adjacent_mines()` berechnet für jede Zelle die Anzahl benachbarter Minen
- **Nachbar-Abfrage:** `get_neighbors()` gibt alle 8 Nachbarzellen zurück

**Design-Entscheidungen:**
- **Lazy Mine Placement:** Minen werden erst beim ersten Klick platziert (verhindert sofortigen Verlust)
- **Mine-Positions-Tracking:** `mine_positions` Set speichert alle Minen-Positionen für schnellen Zugriff
- **Grenzenprüfung:** `get_cell()` prüft Array-Grenzen und gibt `None` für ungültige Positionen zurück

**Mögliche Optimierungen:**
- ✅ Aktuell: Effizient implementiert
- 💡 Potenzial: Caching von Nachbarzellen für sehr große Spielfelder (aktuell nicht nötig bei 20x30)

---

### 2.3 Game-Klasse (`src/minesweeper/game.py`)

**Zweck:** Verwaltet die gesamte Spiellogik und Spielzustand.

**Implementierung:**
- **Spielzustand:** PLAYING, WON, LOST
- **Erster Klick:** Triggert Minen-Platzierung (ausschließlich geklickter Zelle)
- **Aufdecken:** `reveal_cell()` prüft auf Mine, deckt auf, prüft Gewinn
- **Auto-Aufdecken:** `_auto_reveal_safe_neighbors()` deckt automatisch sichere Nachbarn auf (BFS-Algorithmus)
- **Flaggen:** `toggle_flag()` setzt/entfernt Flaggen
- **Schwierigkeitsgrade:** Dynamische Minen-Anzahl basierend auf Schwierigkeit

**Design-Entscheidungen:**
- **Auto-Aufdecken:** BFS (Breadth-First Search) für effizientes Aufdecken von Bereichen mit 0 Minen
- **Erster Klick:** Garantiert, dass erste Zelle sicher ist (keine Mine)
- **State Management:** Klare Zustandsverwaltung mit `GameState` Enumeration

**Mögliche Optimierungen:**
- ✅ Aktuell: Gut implementiert
- 💡 Potenzial: 
  - Timer-Integration (bereits in GUI vorhanden)
  - Highscore-System
  - Hint-System für schwierige Situationen

---

## Phase 3: GUI Implementation

### 3.1 GameBoard Widget (`src/gui/game_board.py`)

**Zweck:** Zeigt das Spielfeld als interaktive Buttons an.

**Implementierung:**
- **Custom Button:** `CellButton` erbt von `QPushButton` mit Signal-System
- **Grid-Layout:** 20x30 Grid von Buttons
- **Interaktion:** Linksklick = Aufdecken, Rechtsklick = Flagge
- **Visualisierung:** Farbcodierung für Zahlen, Icons für Minen/Flaggen

**Design-Entscheidungen:**
- **Signals:** Verwendet PySide6 Signals für lose Kopplung
- **Update-Mechanismus:** `_update_display()` aktualisiert alle Buttons basierend auf Spielzustand

---

### 3.2 MainWindow (`src/gui/main_window.py`)

**Zweck:** Hauptfenster der Anwendung.

**Implementierung:**
- **Menu-Bar:** Schwierigkeitsgrade, Neues Spiel
- **Status-Bar:** Minen-Zähler, Timer
- **Game-Board:** Integriertes Spielfeld
- **Event-Handling:** Gewinn/Verlust-Meldungen

---

## Phase 4: Reinforcement Learning

### 4.1 Environment (`src/reinforcement_learning/environment.py`)

**Zweck:** Gym-ähnlicher Wrapper rund um die Minesweeper-Logik.

**Implementierung (aktuelle Version):**
- **State Representation:** 7 Kanäle (Hidden-, Flag-, Zahlenmasken, Nachbarschaftsdichten und Hinweissumme). Alle Werte liegen in `[-1, 1]`.
- **Action Space:** Standardmäßig `width × height` (Reveal-only, inspiriert durch [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)). Flags können per `--use-flags` wieder zugeschaltet werden.
- **Reward System:** Fortschrittsbasierte Skalierung (`reward_scale = max(1, width*height/100)`), starke Verluststrafe (`-12 * scale`), hoher Gewinnbonus (`+18 * scale`). Guess-Klicks erhalten einen Malus, Frontier-Züge Bonuspunkte.
- **Action Masking:** `get_valid_actions()` liefert boolsche Maske; `get_action_mask()` erzeugt -inf für ungültige Aktionen (wird direkt in den Q-Werten verwendet).

**Tests:** 13 Testfälle (Initialisierung, Reset, Rewards, Masken, Flag-Rewards, usw.)

---

### 4.2 DQN Network (`src/reinforcement_learning/network.py`)

**Zweck:** CNN extrahiert räumliche Muster und gibt Q-Werte für jede erlaubte Aktion zurück.

**Architektur (conv128x4_dense512x2):**
```
Input: (batch, 7, H, W)
├── [Conv2d + BatchNorm + ReLU] × 4   (je 128 Filter, kernel=3, padding=1)
├── AdaptiveAvgPool2d(8 × 8)          (grenzenlos für verschiedene Brettgrößen)
├── Flatten → 128 × 8 × 8 = 8192 Features
├── Linear(8192 → 512) + ReLU + Dropout(0.25)
├── Linear(512 → 512) + ReLU + Dropout(0.25)
└── Linear(512 → num_actions)
```

**Reasoning:**
- 4 tiefe Conv-Blöcke entsprechen dem in [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning) erprobten Setup und verbessern die Frontier-Erkennung.
- Adaptive Pooling sorgt dafür, dass auch 5×5- oder 40×25-Bretter ohne Architekturänderung funktionieren.
- Dropout reduziert Overfitting auf kleinen Boards.

**Tests:** 7 Testfälle (Initialisierung, Vorwärtspass, Gradienten, Parameteranzahl etc.)

---

### 4.3 DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

**Zweck:** Double-DQN-Agent mit Experience Replay, Masking und linearem Explorations-Schedule.

**Komponenten:**
1. **ReplayBuffer:** `deque` mit max. 10k Einträgen, speichert zusätzlich die zulässigen Aktionen des Folgezustands.
2. **Q-/Target-Network:** identische Netze; Target wird alle 100 Trainingsschritte synchronisiert.
3. **Epsilon-Greedy:** Training verwendet einen linearen Scheduler (1.0 → 0.03/0.05/0.10), gesteuert im Trainer.

**Hyperparameter (abhängig vom Brett):**
- `lr`: Basis 0.001, skaliert für kleinere Bretter leicht nach oben
- `gamma`: 0.95 (≤600 Felder) / 0.98 (größer)
- `batch_size`: 32–96
- `loss`: SmoothL1Loss (Huber)
- `optimizer`: Adam
- `target_update`: alle 100 Steps

**Training Process:**
1. Replay-Sampling + Maskierung ungültiger Aktionen
2. Online-Netz liefert `argmax_a Q(s', a)` nur über gültige Aktionen
3. Target-Netz bewertet diese Aktion (Double DQN)
4. TD-Target = Reward + `gamma * Q_target`
5. Backpropagation + Gradient Clipping (`max_norm=1.0`)
6. Zielnetz-Sync alle 100 Steps
7. Epsilon wird nach jeder Episode via `LinearSchedule` gesetzt (kein Multiplikationsrauschen mehr)

**Design-Entscheidungen:**
- **Frontier-Sampling:** Auch bei Exploration werden Züge nahe bekannter Zahlen bevorzugt.
- **Action Masking:** `-1e9` auf ungültigen Aktionen sorgt dafür, dass `argmax` nie auf bereits aufgedeckte Zellen fällt.
- **Greedy Evaluation:** Während des Trainings werden regelmäßig episodenweise Testläufe mit `ε=0` durchgeführt, um echte Leistung zu messen.

**Tests:** 13 Testfälle (ReplayBuffer, Action Selection, Training Step, Save/Load, Environment-Integration)

---

## Tests

### Test-Struktur:

```
tests/
├── minesweeper/
│   ├── test_cell.py      # Cell-Klasse Tests (9 Tests)
│   ├── test_board.py     # Board-Klasse Tests (7 Tests)
│   └── test_game.py      # Game-Klasse Tests (10 Tests)
├── reinforcement_learning/
│   ├── test_environment.py    # Environment Tests (13 Tests)
│   ├── test_network.py        # DQN Network Tests (7 Tests)
│   └── test_dqn_agent.py      # DQN Agent Tests (13 Tests)
└── run_tests.py          # Test-Runner
```

### Test-Statistik:

- **Gesamt:** 57 Tests
- **Minesweeper:** 24 Tests
- **Reinforcement Learning:** 33 Tests
- **Alle Tests:** ✅ Bestanden

### Tests ausführen:

```bash
python tests/run_tests.py
# oder
python -m pytest tests/
```

---

## Zusammenfassung der Design-Entscheidungen

1. **Modulare Struktur:** Klare Trennung von Spiellogik, GUI und RL
2. **Lazy Mine Placement:** Minen werden erst beim ersten Klick platziert
3. **Auto-Aufdecken:** BFS-Algorithmus für benutzerfreundliches Spiel
4. **State Management:** Klare Zustandsverwaltung mit Enumerationen
5. **Signal-basierte GUI:** Lose Kopplung zwischen GUI und Spiellogik
6. **RL Environment:** Gymnasium-ähnliches Interface für Wiederverwendbarkeit

