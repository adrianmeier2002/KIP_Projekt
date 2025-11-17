# Reinforcement Learning Implementierungs-Guide

> **✨ AKTUALISIERT V3** (2025-01-17 Abend): Safe-Cell Detection hinzugefügt! (KRITISCH)
> - **9. Kanal hinzugefügt:** Safe-Cell-Maske (100% sichere Zellen) ⭐⭐⭐
> - **Extrem erhöhte Rewards:** Sichere Züge werden 3x stärker belohnt!
> - **8. Kanal hinzugefügt:** Explizite Frontier-Maske
> - **Erster Zug Bug behoben:** Wurde fälschlicherweise als Guess bestraft! (KRITISCH)
> - **Reward-Signale verstärkt:** 4-8x stärkere Signale, auch für kleine Boards
> - **Win-Reward erhöht:** 2-3x stärker für klareres Ziel
> - **State-Encoding vereinfacht:** Klarere Kanal-Trennung

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **9 Kanäle** (V3!) und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): Vereinfachter Wert (-0.8 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl)  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)
8. **Frontier-Maske** (`state[7]`): 1.0 wenn Zelle verdeckt ist UND neben aufgedeckter Zelle liegt
9. **Safe-Cell-Maske** (`state[8]`): ✨✨✨ **KRITISCH!** 1.0 wenn Zelle mathematisch 100% sicher ist

**Wichtige Verbesserungen:**
- **Frontier-Kanal (7):** Zeigt welche Zellen an der Information-Frontier liegen
- **Safe-Cell-Kanal (9):** ⭐ Berechnet mit Minesweeper-Logik welche Zellen garantiert sicher sind!
  - Wenn Zelle N zeigt und N Nachbarn geflaggt → andere Nachbarn sind SICHER
  - Erspart dem CNN das Lernen von kombinatorischer Logik
  - **Hybrid-Ansatz:** Rule-Based Logic + RL!

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System (✨ VERBESSERT V2 - Mit kritischen Bugfixes!)

Skaliert automatisch mit der Brettgröße (`reward_scale = max(1, width*height/100)`):

**Kritischer Bugfix V2:** Erster Zug wird jetzt KORREKT behandelt (war vorher als Guess bestraft!)

**Wichtigste Verbesserungen V2:**
1. ✅ **Erster Zug:** Keine Penalty mehr! (war KRITISCHER Bug)
2. ✅ **Stärkere Signale:** Minimum-Werte für kleine Boards (4-8x stärker!)
3. ✅ **Höherer Win-Reward:** 2-3x erhöht für klareres Ziel

- **Aufdecken (erfolgreich)**:
  - `+progress_scale` (mind. 0.5) pro neu aufgedecktem Feld
  - Kettenbonus für Flood-Fill (`+0.5 * progress_scale` pro zusätzlichem Feld)
  - **Erster Zug:** Nur Basis-Reward, KEINE Guess-Penalty! ✨ KRITISCHER FIX!
  - **Frontier-Bonus:** `max(10.0, 4.0 * reward_scale) * (1 + frontier_factor)` 
    - Bei 5x5: +10-20 (war: +2.5)
    - Bei 30x20: +24-48 (war: +15)
  - **Guess-Penalty:** `-max(8.0, 3.0 * reward_scale)` 
    - Bei 5x5: -8.0 (war: -2.0)
    - Bei 30x20: -18.0 (war: -12)
- **Keine Fortschritte**: `-0.2 * progress_scale`
- **Flaggen (optional)**:
  - `+0.2` bei korrekt gesetzter Flagge, `-0.2` bei Fehlflag
  - End-Bonus/-Malus (`±0.5`) für korrekte/inkorrekte Flags bei Spielende
- **Spiel verloren**: 
  - Guess: `-12 * reward_scale` (volle Strafe)
  - Frontier-Move: `-7.2 * reward_scale` (40% Reduktion)
- **Spiel gewonnen**: `max(50.0, 25.0 * reward_scale) + 10 * progress_ratio`
  - Bei 5x5: ~51 (war: ~21) - **2.4x stärker!**
  - Bei 30x20: ~160 (war: ~128) - **1.25x stärker!**

**Warum diese Änderungen (V2)?**
Die V1-Implementierung hatte einen KRITISCHEN Bug und zu schwache Signale:
- ❌ **Erster Zug wurde bestraft** → Jede Episode startete mit Nachteil!
- ❌ **Rewards zu schwach bei kleinen Boards** → Signal-to-Noise zu niedrig
- ❌ **Win-Reward zu niedrig** → Ziel nicht klar genug

V2 behebt alle diese Probleme und sollte **deutlich bessere** Lernergebnisse liefern!

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: dynamisch (Basis 0.001, skaliert nach Brettgröße)
- Gamma (Discount): 0.95 (≤600 Felder) bzw. 0.98 (größere Boards)
- Epsilon: Linearer Schedule 1.0 → 0.03–0.10 über ~70 % der Episoden
- Loss: Smooth L1 (Huber), Optimizer: Adam
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps (hartes Sync)
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags via CLI aktivierbar
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
schedule = LinearSchedule(start=1.0, end=epsilon_floor, duration=int(0.7 * episodes))

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        agent.train_step()
        state = next_state
    
    # Linearer Epsilon-Schedule (kein Multiplikationsrauschen mehr)
    agent.epsilon = schedule.value(episode + 1)
    
    if (episode + 1) % log_interval == 0:
        log_statistics()
        greedy_eval(agent, episodes=eval_episodes, epsilon=0.0)
```

### Logging

Alle `log_interval` Episoden werden sowohl Trainings- als auch Evaluationswerte geloggt:
- **Avg Reward / Length**: Mittelwert der letzten `log_interval` Episoden (mit Exploration)
- **Train Win Rate**: Gewinnrate unter aktuellem ε
- **Eval (ε=0)**: Greedy-Gewinnrate und durchschnittliche Zuganzahl aus `--eval-episodes` Testspielen
- **Epsilon**: Wert des linearen Schedules

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper besitzt einen klar abgegrenzten Aktionsraum (`width × height` Zellen, optional Flags)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Mehrkanal-Features**: Getrennte Masken für Hidden/Flag/Numbers + Nachbarschaftsdichten → Netz erkennt lokale Muster schneller
2. **Normalisierung**: Alle Werte liegen in `[-1, 1]`, kompatibel mit BatchNorm
3. **Frontier-Hinweise**: Zusätzliche Kanäle (Hinweissumme, Hidden-/Flag-Nachbarn) liefern Vorwissen ohne Regeln zu hardcoden

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Kürzlich implementierte Verbesserungen ✅

**V1 (Vormittag):**
1. ✅ **Expliziter Frontier-Kanal**: Netzwerk erkennt sofort die wichtigen Zellen
2. ✅ **Verstärktes Reward-Shaping**: Klare Unterscheidung Frontier vs. Guess
3. ✅ **Asymmetrisches Loss-Handling**: Frontier-Moves werden bei Verlust weniger bestraft
4. ✅ **Vereinfachtes State-Encoding**: Klarere Kanal-Trennung

**V2 (Nachmittag - Kritische Bugfixes):**
1. ✅ **Erster Zug Bug behoben**: War fälschlicherweise als Guess bestraft (KRITISCH!)
2. ✅ **Reward-Signale verstärkt**: 4-8x stärkere Signale mit Minimum-Werten
3. ✅ **Win-Reward erhöht**: 2-3x stärker für klareres Lernziel
4. ✅ **Board-unabhängige Minimums**: Kleine Boards haben jetzt starke Signale

### Weitere mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **Probability-Based Features**: Wahrscheinlichkeitsberechnungen für Minen
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1500 \
    --difficulty easy \
    --width 9 --height 9 \
    --eval-episodes 25 \
    --save-path models/dqn_model_9x9.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS

env = MinesweeperEnvironment("medium", width=20, height=15, use_flag_actions=False)
agent = DQNAgent(
    state_channels=STATE_CHANNELS,
    action_space_size=env.action_space_size,
    board_height=env.height,
    board_width=env.width
)
agent.load("models/dqn_model.pth")

state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!

