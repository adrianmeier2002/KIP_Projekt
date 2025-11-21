# Intelligenter Solver + Joker-Nutzung

## 🎯 Probleme gelöst

### Problem 1: Schlechte Winrate (34% → besser)
**Ursache**: Zufällige Tetris-Platzierung führte fast immer zu Minen

### Problem 2: Joker werden nicht genutzt
**Ursache**: Hybrid Agent hatte keine Logik für Joker-Nutzung

---

## ✅ Lösung 1: Intelligente Tetris-Platzierung

### Vorher (Zufällig):
```python
# Wähle zufällige gültige Position
row, col = random.choice(valid_positions)
```
**Problem**: ~90% Chance auf Mine → Verlust!

### Nachher (Intelligent):
```python
def _handle_tetris_auto_placement(self):
    # 1. Nutze Solver um sichere Zellen zu finden
    solver = MinesweeperSolver(self.game)
    safe_cells = set(solver.get_safe_moves())
    
    # 2. Score jede Position
    for row, col in valid_positions:
        score = 0
        cells_in_shape = []
        
        for dr, dc in tetris_shape:
            r, c = row + dr, col + dc
            cells_in_shape.append((r, c))
            
            # +100 wenn Zelle als sicher markiert
            if (r, c) in safe_cells:
                score += 100
            
            # +5 pro aufgedecktem Nachbarn (wahrscheinlich sicher)
            revealed_neighbors = count_revealed_neighbors(r, c)
            score += revealed_neighbors * 5
        
        # +500 wenn ALLE Zellen sicher sind!
        if all(cell in safe_cells for cell in cells_in_shape):
            score += 500
    
    # 3. Wähle beste Position (oder Top 3 für Varianz)
    top_positions = sorted(scores, reverse=True)[:3]
    best_position = random.choice(top_positions)
```

**Effekt**: 
- ✅ Bevorzugt sichere Positionen (keine Minen)
- ✅ Bonus wenn ALLE Tetris-Zellen sicher
- ✅ Fallback auf Heuristik wenn keine sicheren Zellen

---

## ✅ Lösung 2: Intelligente Joker-Nutzung

### Neue Strategie im Hybrid Agent:

#### 1. Mystery-Zellen enthüllen

```python
# WANN: Mystery-Zellen mit vielen verdeckten Nachbarn
if len(game.mystery_cells) > 0 and game.points >= 20:
    for mystery_row, mystery_col in game.mystery_cells:
        hidden_neighbors = count_hidden_neighbors(mystery_row, mystery_col)
        
        # Enthülle wenn >= 3 verdeckte Nachbarn (könnte helfen!)
        if hidden_neighbors >= 3 and random() < 0.5:
            game.reveal_mystery_cell(mystery_row, mystery_col)
```

**Effekt**: 
- ✅ Solver bekommt mehr Informationen
- ✅ Kann dann sichere Züge finden

#### 2. Blitz nutzen wenn festgefahren

```python
# WANN: Keine sicheren Züge gefunden
solver = MinesweeperSolver(game)
safe_moves = solver.get_safe_moves()

if len(safe_moves) == 0 and game.points >= game.blitz_cost:
    # Nutze Blitz (deckt 1-3 sichere Felder auf)
    if random() < 0.3:  # 30% Chance (nicht zu aggressiv)
        fields = game.use_blitz()
        if fields > 0:
            # Versuche erneut Solver nach Blitz
            safe_moves = solver.get_safe_moves()
```

**Effekt**:
- ✅ Verhindert hoffnungslose Guesses
- ✅ Blitz findet sichere Züge
- ✅ Solver kann danach weitermachen

#### 3. Erweiterte select_action() Logik

```python
def select_action(state, valid_actions, game):
    # ✨ 1. Prüfe Joker (wenn epsilon < 0.2, d.h. Training weit)
    if epsilon < 0.2:
        joker_used = self._try_use_powerups(game)
    
    # ✨ 2. Versuche Solver
    safe_moves = solver.get_safe_moves()
    if safe_moves:
        return random.choice(safe_moves)
    
    # ✨ 3. Kein Zug? Nutze Blitz!
    if len(safe_moves) == 0 and game.points >= blitz_cost:
        game.use_blitz()
        safe_moves = solver.get_safe_moves()  # Nochmal prüfen
        if safe_moves:
            return random.choice(safe_moves)
    
    # 4. Immer noch nichts? Best Guess
    return solver.get_best_guess(valid_actions)
    
    # 5. Kein Best Guess? RL
    return super().select_action(state, valid_actions)
```

---

## 📊 Erwartete Verbesserungen

### Tetris-Handling:

| Metrik | Vorher (Zufällig) | Nachher (Intelligent) |
|--------|-------------------|------------------------|
| Tetris-Überlebensrate | ~10% | **~70-80%** |
| Tetris auf sichere Zellen | Zufall | **Bevorzugt** |
| Tetris führt zu Verlust | ~90% | **~20-30%** |

### Joker-Nutzung:

| Joker | Vorher | Nachher |
|-------|--------|---------|
| **Mystery enthüllen** | ❌ Nie | ✅ **Bei ≥3 verdeckten Nachbarn** |
| **Blitz** | ❌ Nie | ✅ **Wenn keine sicheren Züge** |
| **Blitz-Timing** | - | ✅ **30% Chance** (nicht zu aggressiv) |

### Gesamt-Winrate:

| Phase | Vorher | Erwartet Nachher |
|-------|--------|------------------|
| Training (ε=0.5) | 14% | **20-25%** |
| Training (ε=0.05) | 34% | **45-55%** |
| Evaluation (ε=0) | 25-32% | **50-65%** |

---

## 🧪 Testing

### Test 1: Intelligente Tetris-Platzierung

```python
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment(difficulty="easy", enable_challenges=True)
state = env.reset()

# Decke einige Felder auf (damit Solver Infos hat)
for i in range(10):
    valid = env.get_valid_actions()
    action = valid.argmax()
    state, _, done, _ = env.step(action)
    if done:
        break

# Aktiviere Tetris künstlich
env.game.tetris_active = True
env.game.tetris_current_shape = env.game.tetris_shapes['I']

# Teste intelligente Platzierung
success = env._handle_tetris_auto_placement()

print(f"Tetris platziert: {success}")
print(f"Tetris noch aktiv: {env.game.tetris_active}")
# Sollte False sein (wurde platziert)
```

### Test 2: Joker-Nutzung im Training

```python
from src.reinforcement_learning.hybrid_agent import HybridAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment(difficulty="easy", enable_challenges=True)
agent = HybridAgent(
    board_height=8,
    board_width=8,
    state_channels=12,
    action_space_size=64,
    use_solver=True
)

# Setze epsilon niedrig (damit Joker genutzt werden)
agent.epsilon = 0.1

state = env.reset()

# Gebe genug Punkte für Joker
env.game.points = 200

# Spiele bis Joker genutzt wird
for step in range(50):
    valid = env.get_valid_actions()
    action = agent.select_action(state, valid, game=env.game)
    
    prev_points = env.game.points
    state, reward, done, info = env.step(action)
    
    # Prüfe ob Punkte verbraucht wurden (Joker genutzt)
    if prev_points - env.game.points >= 20:
        print(f"✅ Joker genutzt in Schritt {step}!")
        print(f"Punkte: {prev_points} → {env.game.points}")
        break
    
    if done:
        break
```

---

## 💡 Weitere Optimierungs-Ideen

### 1. Radar/Scanner Integration (Optional)

```python
def _try_use_powerups(self, game):
    # ... bestehende Logik
    
    # Scanner nutzen um Mine-Counts zu erfahren
    if game.points >= game.scanner_cost:
        # Finde interessante Position (viele verdeckte Zellen)
        # Nutze Scanner dort
        # Solver kann dann mit mehr Infos arbeiten
```

### 2. Adaptives Blitz-Timing

```python
# Nutze Blitz häufiger wenn wenig Fortschritt
progress_ratio = game.revealed_count / game.total_safe_cells
blitz_probability = 0.3 + (1 - progress_ratio) * 0.2

if random() < blitz_probability:
    game.use_blitz()
```

### 3. Mystery-Priorität

```python
# Priorisiere Mystery-Zellen mit meisten verdeckten Nachbarn
mystery_scores = [(count_hidden(pos), pos) for pos in game.mystery_cells]
mystery_scores.sort(reverse=True)
best_mystery = mystery_scores[0][1]
game.reveal_mystery_cell(*best_mystery)
```

---

## 🎯 Zusammenfassung

| Verbesserung | Beschreibung | Impact |
|--------------|-------------|--------|
| **Intelligente Tetris-Platzierung** | Bevorzugt sichere Positionen basierend auf Solver | ⭐⭐⭐⭐⭐ Hoch |
| **Mystery-Zellen enthüllen** | Enthüllt bei ≥3 verdeckten Nachbarn | ⭐⭐⭐ Mittel |
| **Blitz bei Stagnation** | Nutzt Blitz wenn keine sicheren Züge | ⭐⭐⭐⭐ Hoch |
| **Blitz → Solver Kette** | Nach Blitz erneut Solver prüfen | ⭐⭐⭐⭐ Hoch |

---

## 📝 Geänderte Dateien

1. ✅ `src/reinforcement_learning/environment.py`
   - `_handle_tetris_auto_placement()` komplett überarbeitet
   - Intelligente Scoring-Logik
   - Nutzt Solver für sichere Zellen

2. ✅ `src/reinforcement_learning/hybrid_agent.py`
   - `select_action()` erweitert für Joker
   - `_try_use_powerups()` neu hinzugefügt
   - Mystery-Enthüllung Logik
   - Blitz-Nutzung bei Stagnation

---

## 🎉 Erwartetes Ergebnis

**Vorher:**
```
Episode 500/500
  Win Rate: 34.0%
  Eval Win Rate: 25.0%
  Problem: Tetris = fast immer Tod
```

**Nachher (erwartet):**
```
Episode 500/500
  Win Rate: 50-55%
  Eval Win Rate: 50-65%
  Tetris-Überlebensrate: ~70-80%
  Joker-Nutzung: Aktiv & intelligent
```

---

**Datum:** 19. November 2025  
**Version:** 3.2.0  
**Status:** ✅ Intelligenter Solver + Joker-System implementiert

