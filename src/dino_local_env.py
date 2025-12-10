"""Chrome Dino Gymnasium environment and simple wrappers used in the project."""
from __future__ import annotations
import os, enum
from typing import Any, Tuple, List, Deque
from collections import deque

import numpy as np
from PIL import Image, ImageDraw
import pygame, pygame.freetype
import gymnasium as gym

# -----------------------------
# ENV CONSTANTS (tweakable)
# -----------------------------
WINDOW_SIZE = (1024, 512)     # (w, h)
JUMP_DURATION = 12
JUMP_VEL = 100
OBSTACLE_MIN_CNT = 400
MAX_SPEED = 100
MAX_CACTUS_SPAWN_PROB = 0.7
BASE_CACTUS_SPAWN_PROB = 0.3
BIRD_SPAWN_PROB = 0.3
RENDER_FPS = 15
COLLISION_THRESHOLD = 20
DIFFICULTY_INCREASE_FREQ = 20

# REWARD CONSTANTS 
SURVIVAL_REWARD = 0.1          # reward per frame survived
COLLISION_PENALTY = 1.0        # penalty for collision (simple, proven approach)

ASSETS_DIR = "assets"  # where PNGs live

# -----------------------------
# ENUMS
# -----------------------------
class Action(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2

class DinoState(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2

class GameMode(str, enum.Enum):
    NORMAL = "normal"   # episodic: collision ends episode
    TRAIN = "train"     # shaped: collisions penalize but continue (short cap)

class RenderMode(str, enum.Enum):
    HUMAN = "human"
    RGB = "rgb_array"

# -----------------------------
# ASSETS
# -----------------------------
def _ensure_assets():
    """Create simple placeholder sprites if real PNGs are missing."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    files = {
        "Track.png": (WINDOW_SIZE[0], 64),
        "DinoRun1.png": (88, 94),
        "DinoRun2.png": (88, 94),
        "DinoDuck1.png": (118, 60),
        "DinoDuck2.png": (118, 60),
        "DinoJump.png": (88, 94),
        "LargeCactus1.png": (50, 100),
        "LargeCactus2.png": (75, 100),
        "LargeCactus3.png": (100, 100),
        "SmallCactus1.png": (34, 70),
        "SmallCactus2.png": (68, 70),
        "SmallCactus3.png": (102, 70),
        "Bird1.png": (92, 64),
        "Bird2.png": (92, 64),
    }
    for name, (w, h) in files.items():
        path = os.path.join(ASSETS_DIR, name)
        if not os.path.exists(path):
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            # simple shape placeholders
            if "Track" in name:
                draw.rectangle([0, 0, w, h], fill=(120, 120, 120, 255))
            elif "Bird" in name:
                draw.ellipse([4, 4, w-4, h-4], fill=(80, 80, 80, 255))
            elif "Cactus" in name:
                draw.rectangle([6, 6, w-6, h-6], fill=(50, 160, 50, 255))
            else:  # Dino
                draw.rectangle([6, 6, w-6, h-6], fill=(40, 40, 40, 255))
            img.save(path)

class Assets:
    def __init__(self):
        _ensure_assets()
        load = pygame.image.load
        p = os.path.join
        self.track = load(p(ASSETS_DIR, "Track.png"))
        self.dino_runs = [load(p(ASSETS_DIR,"DinoRun1.png")), load(p(ASSETS_DIR,"DinoRun2.png"))]
        self.dino_ducks = [load(p(ASSETS_DIR,"DinoDuck1.png")), load(p(ASSETS_DIR,"DinoDuck2.png"))]
        self.dino_jump = load(p(ASSETS_DIR,"DinoJump.png"))
        self.cactuses = [
            load(p(ASSETS_DIR,"LargeCactus1.png")),
            load(p(ASSETS_DIR,"LargeCactus2.png")),
            load(p(ASSETS_DIR,"LargeCactus3.png")),
            load(p(ASSETS_DIR,"SmallCactus1.png")),
            load(p(ASSETS_DIR,"SmallCactus2.png")),
            load(p(ASSETS_DIR,"SmallCactus3.png")),
        ]
        self.birds = [load(p(ASSETS_DIR,"Bird1.png")), load(p(ASSETS_DIR,"Bird2.png"))]

# -----------------------------
# OBJECTS
# -----------------------------
class EnvObject:
    rect: pygame.Rect
    def step(self, *args, **kwargs): ...
    def render(self, canvas: pygame.Surface, *args, **kwargs): ...

class Obstacle(EnvObject):
    needs_collision_check = True
    def collide(self, o: pygame.Rect) -> bool:
        return self.rect.colliderect(
            o.left + COLLISION_THRESHOLD,
            o.top + COLLISION_THRESHOLD,
            o.width - 2 * COLLISION_THRESHOLD,
            o.height - 2 * COLLISION_THRESHOLD,
        )
    def is_inside(self) -> bool: return False

class Bird(Obstacle):
    def __init__(self, assets: Assets):
        self._assets = assets.birds
        self.rect = self._assets[0].get_rect()
        self.rect.x = WINDOW_SIZE[0]
        self.rect.y = 360
    def step(self, speed: int):
        self.rect.x -= speed
        self._assets[0], self._assets[1] = self._assets[1], self._assets[0]
    def is_inside(self) -> bool: return self.rect.x + self._assets[0].get_width() > 0
    def render(self, canvas: pygame.Surface): canvas.blit(self._assets[0], self.rect)

class Cactus(Obstacle):
    def __init__(self, assets: Assets, id: int):
        self._asset = assets.cactuses[id]
        self.rect = self._asset.get_rect()
        self.rect.x = WINDOW_SIZE[0]
        self.rect.y = WINDOW_SIZE[1] - self._asset.get_height() - 7
    def step(self, speed: int): self.rect.x -= speed
    def is_inside(self) -> bool: return self.rect.x + self._asset.get_width() > 0
    def render(self, canvas: pygame.Surface): canvas.blit(self._asset, self.rect)

class Dino(EnvObject):
    def __init__(self, assets: Assets):
        self._run_assets = assets.dino_runs
        self._duck_assets = assets.dino_ducks
        self._jump_asset = assets.dino_jump
        self._jump_timer = 0
        self.state = DinoState.STAND
    def step(self, action: Action):
        self._run_assets[0], self._run_assets[1] = self._run_assets[1], self._run_assets[0]
        self._duck_assets[0], self._duck_assets[1] = self._duck_assets[1], self._duck_assets[0]
        if self.state == DinoState.JUMP:
            self._jump_timer -= 1
            if self._jump_timer < 0: self.state = DinoState.STAND
        if self.state != DinoState.JUMP:
            if action == Action.STAND: self.state = DinoState.STAND
            elif action == Action.JUMP: self.state = DinoState.JUMP; self._jump_timer = JUMP_DURATION
            else: self.state = DinoState.DUCK
    def get_data(self) -> Tuple[pygame.Surface, pygame.Rect]:
        if self.state == DinoState.STAND:
            asset = self._run_assets[0]; y = WINDOW_SIZE[1] - asset.get_height()
        elif self.state == DinoState.JUMP:
            asset = self._jump_asset; y = WINDOW_SIZE[1] - self._get_jump_offset() - asset.get_height()
        else:
            asset = self._duck_assets[0]; y = WINDOW_SIZE[1] - asset.get_height()
        rect = pygame.Rect(50, y, asset.get_width(), asset.get_height())
        return asset, rect
    def _get_jump_offset(self) -> int:
        a = -JUMP_VEL / (JUMP_DURATION / 2); t = JUMP_DURATION - self._jump_timer
        return int(JUMP_VEL * t + 0.5 * a * (t**2))
    def render(self, canvas: pygame.Surface):
        asset, rect = self.get_data(); canvas.blit(asset, rect)

class Track(EnvObject):
    def __init__(self, assets: Assets):
        self._asset = assets.track; self._track_offset_x = 0
        self._track_w = self._asset.get_width(); self._track_h = self._asset.get_height()
    def step(self, speed: int): self._track_offset_x -= speed
    def render(self, canvas: pygame.Surface):
        canvas.blit(self._asset, (self._track_offset_x, WINDOW_SIZE[1] - self._track_h))
        if self._track_offset_x + self._track_w < WINDOW_SIZE[0]:
            start_x = self._track_offset_x + self._track_w - 10
            canvas.blit(self._asset, (start_x, WINDOW_SIZE[1] - self._track_h))
            if start_x <= 0: self._track_offset_x = start_x

# -----------------------------
# GYMNASIUM ENV
# -----------------------------
class DinoEnv(gym.Env):
    metadata = {"render_fps": RENDER_FPS, "render_modes": [RenderMode.HUMAN, RenderMode.RGB]}
    def __init__(self, render_mode: RenderMode|None, game_mode: GameMode = GameMode.NORMAL, train_frame_limit=500):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(len(list(Action)))
        self.observation_space = gym.spaces.Box(0, 255, shape=(WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8)
        self._game_mode = game_mode; self._train_frame_limit = train_frame_limit
        self._window = None; self._clock = None
        pygame.freetype.init()
        self._game_font = pygame.freetype.SysFont(pygame.freetype.get_default_font(), 24)
        if self.render_mode == RenderMode.HUMAN:
            pygame.init(); pygame.display.init()
            self._window = pygame.display.set_mode(WINDOW_SIZE)
            self._clock = pygame.time.Clock()
        self._init_game_data()

    def _init_game_data(self):
        self._assets = Assets()
        self._frame = 0; self._speed = 20; self._spawn_prob = BASE_CACTUS_SPAWN_PROB
        self._obstacle_cnt = OBSTACLE_MIN_CNT
        self._track = Track(self._assets)
        self._agent = Dino(self._assets)
        self._obstacles: List[Obstacle] = []

    def reset(self, seed: int|None=None, options: dict[str, Any]|None=None):
        super().reset(seed=seed, options=options)
        self._init_game_data()
        return self._render_frame(), {}

    def step(self, action: Action):
        terminated = False
        reward = 0.0  # Start with zero, add rewards as events occur
        
        self._frame += 1 
        self._obstacle_cnt += self._speed
        
        if self._frame % DIFFICULTY_INCREASE_FREQ == 0:
            self._speed = min(MAX_SPEED, self._speed + 1)
            self._spawn_prob = min(MAX_CACTUS_SPAWN_PROB, self._spawn_prob * 1.01)
        
        self._track.step(self._speed)
        self._agent.step(action)
        for o in self._obstacles:
            o.step(self._speed)
        self._obstacles = [o for o in self._obstacles if o.is_inside()]
        
        _, agent_rect = self._agent.get_data()
        for o in self._obstacles:
            if not o.needs_collision_check:
                continue
            if o.collide(agent_rect):
                o.needs_collision_check = False
                reward -= COLLISION_PENALTY  # -1.0 penalty for collision
                if self._game_mode == GameMode.NORMAL:
                    terminated = True
            else:
                # Agent passes an obstacle without colliding - give success reward
                if agent_rect.left > o.rect.right:
                    o.needs_collision_check = False
                    reward += 1.0  # Clear success signal
        
        if self._game_mode == GameMode.TRAIN and self._frame >= self._train_frame_limit:
            terminated = True
        
        self._spawn_obstacle_maybe()
        return self._render_frame(), reward, terminated, False, {}

    def _spawn_obstacle_maybe(self):
        # Add randomization to spawn timing (+ - 20% variance)
        min_gap = max(OBSTACLE_MIN_CNT, JUMP_DURATION * self._speed)
        random_gap = min_gap + self.np_random.integers(-int(min_gap * 0.2), int(min_gap * 0.2))
        
        if self._obstacle_cnt > random_gap:
            # Increase bird spawn probability for more variety
            bird_prob = min(0.3, BIRD_SPAWN_PROB * (1 + self._speed / 200.0))
            
            if self.np_random.choice(2, 1, p=[1 - self._spawn_prob, self._spawn_prob])[0]:
                # Random cactus type with equal probability
                id = self.np_random.choice(6, 1)[0]
                self._obstacles.append(Cactus(self._assets, int(id)))
            elif self.np_random.choice(2, 1, p=[1 - bird_prob, bird_prob])[0]:
                self._obstacles.append(Bird(self._assets))
            self._obstacle_cnt = 0

    def render(self):
        if self.render_mode == RenderMode.RGB:
            return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        canvas = pygame.Surface(WINDOW_SIZE); canvas.fill((255, 255, 255))
        self._track.render(canvas); self._agent.render(canvas)
        for o in self._obstacles: o.render(canvas)
        text_surface, _ = self._game_font.render(f"score: {self._frame}", (0, 0, 0))
        canvas.blit(text_surface, (10, 10))
        if self._window is not None and self._clock is not None:
            self._window.blit(canvas, canvas.get_rect()); pygame.event.pump(); pygame.display.update(); self._clock.tick(self.metadata["render_fps"])
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))

    def close(self):
        if self._window is not None:
            pygame.display.quit(); pygame.quit()

# -----------------------------
# LIGHTWEIGHT WRAPPERS (local)
# -----------------------------
class GrayResizeObs(gym.ObservationWrapper):
    """Convert RGB -> grayscale (L) and resize to (84,84); keep_dim=True => (84,84,1)"""
    def __init__(self, env: gym.Env, size=(84,84)):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(self.size[1], self.size[0], 1), dtype=np.uint8
        )
    def observation(self, obs):
        img = Image.fromarray(obs).convert("L").resize(self.size)
        arr = np.array(img, dtype=np.uint8)
        return np.expand_dims(arr, axis=2)  # (H,W,1)

class FrameStackK(gym.Wrapper):
    """Stack last k grayscale frames along channel dimension -> (H,W,k)"""
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        h, w, c = env.observation_space.shape
        assert c == 1, "FrameStackK expects grayscale channel=1 from GrayResizeObs"
        self.k = k
        self.stack: Deque[np.ndarray] = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(h, w, k), dtype=np.uint8
        )
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.stack.clear()
        for _ in range(self.k):
            self.stack.append(obs)
        return self._get_obs(), info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.stack.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    def _get_obs(self):
        return np.concatenate(list(self.stack), axis=2)  # (H,W,k)

# -----------------------------
# HELPERS FOR TRAIN/EVAL
# -----------------------------
def make_dino_env(train_mode: bool = True, stack_k: int = 4):
    """
    Returns a Gymnasium env wrapped for pixel RL:
      - RGB frames from DinoEnv
      - GrayResizeObs -> (84,84,1)
      - FrameStackK(k) -> (84,84,k)
    For SB3 CNN later, we will add VecTransposeImage to convert to CHW.
    """
    base = DinoEnv(
        render_mode=RenderMode.RGB,
        game_mode=GameMode.TRAIN if train_mode else GameMode.NORMAL,
        train_frame_limit=600,
    )
    env = GrayResizeObs(base, size=(84,84))
    env = FrameStackK(env, k=stack_k)
    return env
