# Advanced Features Training Guide

## 🎯 Überblick

Dieses Dokument beschreibt, wie das **Reinforcement Learning Model** verbessert wurde, um **mit den neuen Features** (Mystery-Zellen, Speed-Felder, Tetris-Felder) umzugehen.

## 🔥 Philosophie

**Das Modell wird MIT den Features trainiert, nicht ohne!**

Die neuen Features sind Teil der Herausforderung. Anstatt sie zu deaktivieren, haben wir:
1. ✅ State-Representation erweitert (3 neue Channels)
2. ✅ Reward-System angepasst
3. ✅ Feature-Frequenzen optimiert (aber AKTIV!)

---

## 🧠 State-Representation (12 Channels)

### Erweiterte Channels:

| Channel | Beschreibung | Zweck |
|---------|-------------|-------|
| 0 | Basis-Encoding | Allgemeine Zellinformation |
| 1 | Hidden-Maske | Verdeckte Zellen |
| 2 | Flag-Maske | Geflaggte Zellen |
| 3 | Aufgedeckte Zahl | Mine-Anzahl |
| 4 | Verdeckte Nachbarn | Nachbarschaft |
| 5 | Geflaggte Nachbarn | Nachbarschaft |
| 6 | Hinweis-Summe | Nachbarschaft |
| 7 | Frontier-Maske | Grenz-Zellen |
| 8 | Safe-Cell-Maske | 100% sichere Zellen |
| **9** | **Mystery-Maske** ✨ | **Mystery-Zellen** |
| **10** | **Speed-Aktiv** ✨ | **Speed-Timer aktiv** |
| **11** | **Tetris-Aktiv** ✨ | **Tetris-Modus aktiv** |

### Implementierung:

```python
# Channel 9: Mystery-Zellen
for mystery_pos in self.game.mystery_cells:
    r, c = mystery_pos
    state[9, r, c] = 1.0

# Channel 10: Speed-Timer (globaler Zustand)
if self.game.speed_active:
    state[10, :, :] = self.game.speed_time_remaining / self.game.speed_time_limit

# Channel 11: Tetris-Modus (globaler Zustand)
if self.game.tetris_active:
    state[11, :, :] = 1.0
```

---

## 🎁 Reward-System Anpassungen

### 1. Gewinn-Bonus mit Challenges

Wenn das Spiel mit aktiven Challenges gewonnen wird:

```python
if self.game.is_won():
    win_reward = max(50.0, 25.0 * board_scale) + 10.0 * progress_ratio
    
    # BONUS für Challenges!
    if len(self.game.mystery_cells) > 0 or len(self.game.speed_cells) > 0:
        challenge_bonus = 10.0 * board_scale
        win_reward += challenge_bonus
```

**Effekt**: Agent wird ermutigt, trotz Challenges zu gewinnen!

### 2. Speed-Challenge Belohnungen

```python
# Bonus für Speed-Timer bewältigen
if self.game.speed_active:
    speed_bonus = 2.0 * self.progress_scale
    total_reward += speed_bonus

# Reduzierte Strafe bei Speed-Timeout
if self.game.speed_active and self.game.speed_time_remaining <= 0:
    return base_penalty * 0.5  # Nur 50% Strafe
```

**Effekt**: Agent lernt, unter Zeitdruck zu handeln!

### 3. Mystery-Zellen Handling

Mystery-Zellen werden **sichtbar** im State (Channel 9), aber der Solver ignoriert sie.

**Effekt**: Agent lernt, Mystery-Zellen zu identifizieren und zu vermeiden (oder enthüllen, wenn nötig)!

---

## ⚙️ Feature-Frequenzen (Optimiert)

Die Frequenzen wurden **reduziert aber nicht deaktiviert**:

| Feature | Alte Frequenz | Neue Frequenz | Änderung |
|---------|--------------|---------------|----------|
| Mystery ❓ | 4-8 Felder | **15-25 Felder** | 3x seltener |
| Speed ⚡ | 6-12 Felder | **20-30 Felder** | 3x seltener |
| Tetris 🎮 | 10-15 Felder | **25-40 Felder** | 2.5x seltener |

**Warum reduzieren?**
- Zu häufig → Agent kann nicht lernen
- Zu selten → Kein Training für Features
- **Optimiert** → Balance zwischen Lernen & Challenge!

---

## 🚀 Training-Strategie

### Phase 1: Curriculum Learning (Empfohlen!)

1. **Früh-Training** (enable_challenges=False):
   - 0-1000 Episoden ohne Challenges
   - Agent lernt Basis-Minesweeper
   
2. **Übergangs-Training** (reduzierte Frequenzen):
   - 1000-3000 Episoden mit Challenges (aktuelle Frequenzen)
   - Agent lernt Feature-Handling
   
3. **Experten-Training** (normale Frequenzen):
   - 3000+ Episoden mit Challenges
   - Agent meistert alle Features

### Phase 2: Standard-Training (enable_challenges=True)

```python
# EMPFOHLEN: Mit Challenges trainieren!
env = MinesweeperEnvironment(
    difficulty="easy",
    width=10,
    height=10,
    enable_challenges=True  # ⬅️ Features AKTIV!
)

# Training-Loop
for episode in range(5000):
    state = env.reset()
    # ... Training
```

**Das Modell wird mit den optimierten Frequenzen trainiert!**

---

## 📊 Erwartete Performance

### Mit 12-Channel State + Reward-Anpassungen:

| Metrik | Ohne Optimierung | Mit Optimierung | Verbesserung |
|--------|------------------|-----------------|--------------|
| **Winrate** | 10-20% | **40-60%** | +3x |
| **Solver Usage** | 20-30% | **60-80%** | +3x |
| **Avg. Episode Length** | 50-100 | **30-60** | Effizienter |
| **Challenge Completion** | 0-10% | **30-50%** | Deutlich besser |

---

## 🧪 Testing

### Test 1: State Repräsentation

```python
env = MinesweeperEnvironment(difficulty="easy", enable_challenges=True)
state = env.reset()

print(f"State Shape: {state.shape}")  # Sollte (12, height, width) sein
print(f"Channel 9 (Mystery): {state[9].sum()}")
print(f"Channel 10 (Speed): {state[10].sum()}")
print(f"Channel 11 (Tetris): {state[11].sum()}")
```

### Test 2: Reward-System

```python
# Simuliere Speed-Challenge
env.game.speed_active = True
env.game.speed_time_remaining = 3.0
env.game.reveal_cell(3, 3)

# Reward sollte Speed-Bonus enthalten
state, reward, done, info = env.step(action)
print(f"Reward mit Speed-Bonus: {reward}")
```

### Test 3: Feature-Frequenzen

```python
game = Game(difficulty="easy", enable_challenges=True)
game.board.place_mines(5, 5)
game.first_click = False

# Decke 50 Felder auf
for i in range(50):
    # ... reveal cells

total_challenges = len(game.mystery_cells) + len(game.speed_cells) + len(game.tetris_cells)
print(f"Challenges nach 50 Feldern: {total_challenges}")  # Sollte ~3-5 sein
```

---

## 💡 Best Practices

### 1. Netzwerk-Architektur anpassen

Das Netzwerk muss 12 Input-Channels verarbeiten:

```python
from src.reinforcement_learning.network import DQNNetwork

network = DQNNetwork(
    input_channels=12,  # ⬅️ WICHTIG: 12 statt 9!
    num_actions=width * height,
    board_height=height,
    board_width=width
)
```

### 2. Training-Parameter

```python
agent = HybridAgent(
    board_height=10,
    board_width=10,
    state_channels=12,  # ⬅️ WICHTIG!
    action_space_size=100,
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995
)
```

### 3. Training-Loop

```python
for episode in range(5000):
    state = env.reset()
    total_reward = 0
    
    while not env.game.is_game_over():
        # Nutze Hybrid Agent (Solver + RL)
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, game=env.game)
        
        next_state, reward, done, info = env.step(action)
        
        # Training step
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Statistiken
    if episode % 100 == 0:
        stats = agent.get_stats()
        print(f"Episode {episode}: Reward={total_reward:.1f}, "
              f"Solver={stats['solver_percentage']:.1f}%, "
              f"Won={env.game.is_won()}")
```

---

## 🎯 Ziel-Performance

Nach 5000 Episoden Training:

- ✅ **60%+ Winrate** (mit Challenges!)
- ✅ **70%+ Solver Usage**
- ✅ **Kann Mystery-Zellen identifizieren**
- ✅ **Bewältigt Speed-Challenges**
- ✅ **Lernt Tetris-Modus**

---

## 📝 Zusammenfassung

| Was wurde geändert | Wie | Effekt |
|-------------------|-----|--------|
| **State-Channels** | +3 Channels (Mystery/Speed/Tetris) | Agent sieht Features |
| **Reward-System** | Bonusse für Challenges | Agent lernt Feature-Handling |
| **Feature-Frequenzen** | 2.5-3x seltener | Lernbar aber herausfordernd |
| **Solver-Integration** | Ignoriert Mystery-Zellen korrekt | Faire Hilfe |

---

## 🎉 Ergebnis

**Das Modell kann jetzt MIT den Features trainiert werden!**

- ✅ Features sind Teil des Trainings
- ✅ Optimierte Lernkurve
- ✅ Herausfordernd aber fair
- ✅ Bessere Gesamt-Performance

---

**Datum:** 19. November 2025  
**Version:** 3.0.0 - Advanced Features Integration  
**Status:** ✅ Produktionsbereit für Training

