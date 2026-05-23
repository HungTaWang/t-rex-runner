"""
Chrome Dino — Dueling DDQN + Prioritized Experience Replay + N-step Returns
環境: wayou/T-Rex-Runner 本地伺服器
狀態: JS 直接讀取障礙物座標 (10 維向量)
動作: 0=不動, 1=跳躍
"""

import os, time, random, math, threading
from collections import deque
import http.server, socketserver

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"運算設備: {device}")

# ─────────────────────────────────────────────────────────────────
# 本地 HTTP 伺服器 (wayou T-Rex-Runner)
# 請先 git clone https://github.com/wayou/t-rex-runner.git
# 到 GAME_DIR 指定的路徑
# ─────────────────────────────────────────────────────────────────
GAME_DIR = os.path.join(os.path.dirname(__file__), "t-rex-runner")
GAME_PORT = 8787

def _start_local_server():
    """在背景執行 wayou T-Rex-Runner 靜態伺服器。"""
    if not os.path.isdir(GAME_DIR):
        print(f"[警告] 找不到 {GAME_DIR}，改用 chrome://dino")
        return False

    import functools

    # 用 partial 正確傳入 directory，避免污染 class 屬性
    class SilentHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args): pass  # 靜默 HTTP log

    Handler = functools.partial(SilentHandler, directory=GAME_DIR)

    # 覆寫 handle_error：忽略 Windows 的 ConnectionResetError (WinError 10054)
    # 這是 Chrome 關閉連線時的正常行為，不是真正的錯誤
    class QuietTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
        def handle_error(self, request, client_address):
            import sys
            exc = sys.exc_info()[1]
            if isinstance(exc, ConnectionResetError):
                return   # 靜默忽略
            super().handle_error(request, client_address)

    try:
        httpd = QuietTCPServer(("", GAME_PORT), Handler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"[伺服器] wayou T-Rex-Runner 於 http://localhost:{GAME_PORT}")
        return True
    except OSError:
        print(f"[伺服器] Port {GAME_PORT} 已在使用中，假設伺服器已啟動。")
        return True

# ─────────────────────────────────────────────────────────────────
# 狀態特徵 (JS 直接讀取，無需影像辨識)
# ─────────────────────────────────────────────────────────────────
# 狀態向量 (10 維，已正規化):
#   [speed, dino_y,
#    dist1, y1, w1, h1,   # 第 1 個障礙物
#    dist2, y2, w2, h2]   # 第 2 個障礙物 (若無則補 0)
# ─────────────────────────────────────────────────────────────────
_JS_GET_STATE = """
return (function(){
    var r = Runner.instance_;
    var obs = r.horizon.obstacles;
    var tx  = r.tRex.xPos;
    var ty  = r.tRex.yPos;
    var spd = r.currentSpeed || 6;
    var out = [spd / 13.0, ty / 150.0];
    for (var i = 0; i < 2; i++) {
        if (i < obs.length) {
            var o = obs[i];
            out.push((o.xPos - tx) / 600.0);
            out.push(o.yPos / 150.0);
            out.push(o.width / 100.0);
            out.push((o.typeConfig ? o.typeConfig.height : 50) / 100.0);
        } else {
            out.push(1.0, 0.0, 0.0, 0.5);  // 無障礙物的預設填充
        }
    }
    return out;
})();
"""

STATE_DIM = 10  # 與上方輸出對應

# ─────────────────────────────────────────────────────────────────
# 環境
# ─────────────────────────────────────────────────────────────────
class DinoEnv:
    def __init__(self, use_local: bool = True):
        opts = Options()
        opts.add_argument("--mute-audio")
        opts.add_argument("--disable-web-security")
        self.driver = webdriver.Chrome(options=opts)

        if use_local:
            self.driver.get(f"http://localhost:{GAME_PORT}/index.html")
        else:
            self.driver.get("https://chromedino.com/")
        time.sleep(2)

        # 啟動遊戲
        try:
            self.driver.execute_script("Runner.instance_.playIntro()")
        except Exception:
            self.driver.execute_script(
                "document.dispatchEvent(new KeyboardEvent('keydown',{'keyCode':32,'which':32}));"
            )
        time.sleep(0.8)

    # ── JS 狀態讀取 ───────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        raw = self.driver.execute_script(_JS_GET_STATE)
        return np.array(raw, dtype=np.float32)

    # ── 重設 ──────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        self.driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.15)
        return self._get_state()

    # ── 動作對應按鍵 ─────────────────────────────────────────────
    _JS_UP_DOWN = "document.dispatchEvent(new KeyboardEvent('keydown',{'keyCode':38,'which':38}));"
    _JS_DN_UP   = "document.dispatchEvent(new KeyboardEvent('keyup', {'keyCode':40,'which':40}));"

    def step(self, action: int):
        if action == 1:  # 跳躍
            self.driver.execute_script(self._JS_DN_UP)
            self.driver.execute_script(self._JS_UP_DOWN)
        # action == 0: 不動（不送任何按鍵）

        time.sleep(0.05)

        crashed = self.driver.execute_script("return Runner.instance_.crashed")
        speed   = self.driver.execute_script("return Runner.instance_.currentSpeed") or 6.0
        reward  = -5.0 if crashed else 0.15 + (speed - 6.0) * 0.05  # 降低懲罰 / 提高存活+速度獎勵

        next_state = self._get_state()
        return next_state, reward, bool(crashed)

    def close(self):
        try: self.driver.quit()
        except: pass


# ─────────────────────────────────────────────────────────────────
# Dueling DQN — MLP 版 (輸入為特徵向量，非影像)
# ─────────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        # ── 共享主幹 ──
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        # ── Value 分支 ──
        self.value = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        # ── Advantage 分支 ──
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.shared(x)               # (B, hidden)
        v = self.value(f)                # (B, 1)
        a = self.advantage(f)            # (B, A)
        return v + (a - a.mean(dim=1, keepdim=True))


# ─────────────────────────────────────────────────────────────────
# Prioritized Experience Replay (SumTree)
# ─────────────────────────────────────────────────────────────────
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree  = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data  = [None] * capacity
        self.write = 0
        self.n     = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent: self._propagate(parent, change)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def _retrieve(self, idx, s):
        left, right = 2*idx+1, 2*idx+2
        if left >= len(self.tree): return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx  = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self): return self.tree[0]


class PrioritizedReplayBuffer:
    ALPHA = 0.3   # 優先度指數（降低：避免長回合 transitions 壟斷 buffer）
    BETA0 = 0.5   # IS 權重初始值（稍高：更快矯正優先度偏差）
    BETA_FRAMES = 500_000
    EPS   = 1e-6

    def __init__(self, capacity: int):
        self.tree   = SumTree(capacity)
        self.frame  = 1
        self._max_p = 1.0

    def push(self, *transition):
        self.tree.add(self._max_p, transition)

    def sample(self, batch_size: int):
        idxs, priorities, batch = [], [], []
        seg = self.tree.total / batch_size
        for i in range(batch_size):
            # 重試直到取得非 None 的資料（避免命中未初始化的葉節點）
            for _ in range(20):
                s   = random.uniform(seg*i, seg*(i+1))
                idx, p, data = self.tree.get(s)
                if data is not None:
                    break
            else:
                # 萬一真的找不到，從整棵樹隨機取
                s   = random.uniform(0, self.tree.total)
                idx, p, data = self.tree.get(s)
            idxs.append(idx); priorities.append(p); batch.append(data)

        beta  = min(1.0, self.BETA0 + self.frame * (1.0 - self.BETA0) / self.BETA_FRAMES)
        self.frame += 1
        # 以 EPS 作地板，防止 0 優先度造成 inf IS 權重
        probs = np.maximum(np.array(priorities, dtype=np.float32), self.EPS) / (self.tree.total + self.EPS)
        weights = (self.tree.n * probs) ** (-beta)
        weights /= weights.max()

        s, a, r, ns, d = zip(*batch)
        # 狀態為 1D 向量，用 np.stack 而非 np.concatenate
        return (np.stack(s), a, r, np.stack(ns), d,
                idxs, weights.astype(np.float32))

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(float(err)) + self.EPS) ** self.ALPHA
            self._max_p = max(self._max_p, p)
            self.tree.update(idx, p)

    def __len__(self): return self.tree.n


# ─────────────────────────────────────────────────────────────────
# N-step Return Buffer
# ─────────────────────────────────────────────────────────────────
class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.buf = deque()

    def push(self, transition):
        self.buf.append(transition)

    def ready(self): return len(self.buf) >= self.n

    def pop(self):
        """計算 n-step 折扣回報，回傳修正後的 transition。"""
        s, a, _, _, _ = self.buf[0]
        _, _, _, ns, d = self.buf[-1]
        R = 0.0
        for i, (_, _, r, _, done_i) in enumerate(self.buf):
            R += (self.gamma ** i) * r
            if done_i: break
        self.buf.popleft()
        return s, a, R, ns, d


# ─────────────────────────────────────────────────────────────────
# 軟更新 (Polyak)
# ─────────────────────────────────────────────────────────────────
def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ─────────────────────────────────────────────────────────────────
# 主訓練迴圈
# ─────────────────────────────────────────────────────────────────
def train(load_model_path=None, save_model_path="dino_ddqn_dueling.pth"):
    # ── 超參數 ──────────────────────────────────────────────────
    BATCH_SIZE      = 128      # 更穩定的梯度估計
    GAMMA           = 0.99
    N_STEP          = 3        # 3-step 較不稀釋 crash 懲罰，信用歸因更精準
    EPS_START       = 1e-15
    EPS_END         = 0.000    # 保留更多隨機性
    EPS_DECAY       = 15_000  # 更慢衰減，給予更充分的探索
    MEMORY_SIZE     = 200_000  # 加大 buffer，稀釋單一長回合的比例
    MIN_REPLAY_SIZE = 5_000   # 更充分的初始探索再開始學習
    NUM_EPISODES    = 3000
    LR              = 1e-5    # 搭配 scheduler，早期學習更快
    TAU             = 0.001   # 軟更新係數（降低：target net 更穩定，減少 Q 值震盪）
    UPDATE_FREQ     = 4       # 每 4 步才做一次梯度更新，防止長回合壟斷訓練
    SAVE_EVERY      = 100      # 每 N 回合存一次
    LOADED_EPS      = 1e-15

    num_actions = 2   # 0=不動, 1=跳躍

    # ── 啟動本地伺服器 ──────────────────────────────────────────
    use_local = _start_local_server()
    env = DinoEnv(use_local=use_local)

    # ── 網路 ────────────────────────────────────────────────────
    policy_net = DuelingDQN(STATE_DIM, num_actions).to(device)
    target_net = DuelingDQN(STATE_DIM, num_actions).to(device)

    steps_done = 0

    if load_model_path and os.path.exists(load_model_path):
        print(f"載入模型: {load_model_path}")
        policy_net.load_state_dict(torch.load(load_model_path, map_location=device))
        # ✅ 修正：只偏移 steps_done，不改 EPS_START
        # 讓 eps = EPS_END + (EPS_START - EPS_END)*exp(-steps/EPS_DECAY) 在起點等於 LOADED_EPS
        steps_done = int(-EPS_DECAY * math.log(
            max((LOADED_EPS - EPS_END) / (EPS_START - EPS_END), 1e-15)
        ))
        print(f"探索率從 {LOADED_EPS:.4f} 繼續（steps_done 偏移至: {steps_done}）")
        # ✅ 修正：繼續訓練時 buffer 從空開始，用較小的暖身門檻
        MIN_REPLAY_SIZE = 1_000
        print(f"Fine-tune 模式：MIN_REPLAY_SIZE 調整為 {MIN_REPLAY_SIZE}")
    else:
        print("全新訓練。")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    # 每 800 回合 LR * 0.7，衰減更慢，後期仍能持續學習
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.7)
    memory    = PrioritizedReplayBuffer(MEMORY_SIZE)
    n_buf     = NStepBuffer(N_STEP, GAMMA)
    best_reward = -float('inf')  # 用於自動存檔歷史最高分模型
    best_model_path = save_model_path.replace('.pth', '_best.pth')

    try:
        for episode in range(NUM_EPISODES):
            state        = env.reset()
            total_reward = 0.0
            done         = False

            while not done:
                # ε-greedy
                eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)

                if random.random() > eps:
                    with torch.no_grad():
                        st = torch.FloatTensor(state).unsqueeze(0).to(device)
                        action = policy_net(st).argmax(1).item()
                else:
                    # 真正均勻隨機，避免 buffer 被過多「不必要跳躍」污染
                    action = random.randint(0, 1)

                prev_dist1 = state[2]  # 第1個障礙物距離（正規化後，用於偵測是否通過）
                steps_done += 1
                next_state, reward, done = env.step(action)

                # 🆕 成功通過障礙物額外獎勵
                # 前一步障礙物很近(dist<0.15) 且下一步距離跳回遠處(>0.5) → 確認通過
                if not done and prev_dist1 < 0.15 and next_state[2] > 0.5:
                    reward += 3.0

                # 懲罰無意義跳躍：跳躍時最近障礙物還很遠 → 明顯扣分
                # prev_dist1 > 0.4 代表障礙物距離 > 240px，根本不需要跳
                if action == 1 and not done and prev_dist1 > 0.4:
                    reward -= 1.2  # 加重懲罰，讓模型真正在意亂跳的代價

                total_reward += reward

                # n-step buffer（狀態為 1D 向量，不需 expand_dims）
                n_buf.push((state, action, reward, next_state, done))
                if n_buf.ready():
                    memory.push(*n_buf.pop())

                state = next_state

                # ── 學習：每 UPDATE_FREQ 步才更新一次 ────────────────────
                if len(memory) >= MIN_REPLAY_SIZE and steps_done % UPDATE_FREQ == 0:
                    s, a, r_, ns, d_, idxs, w = memory.sample(BATCH_SIZE)

                    # 狀態現在是 (B, STATE_DIM) 的 2D 向量
                    S  = torch.FloatTensor(np.array(s)).to(device)
                    A  = torch.LongTensor(list(a)).unsqueeze(1).to(device)
                    R  = torch.FloatTensor(list(r_)).unsqueeze(1).to(device)
                    NS = torch.FloatTensor(np.array(ns)).to(device)
                    D  = torch.FloatTensor(list(d_)).unsqueeze(1).to(device)
                    W  = torch.FloatTensor(w).unsqueeze(1).to(device)

                    # DDQN: policy_net 選動作, target_net 評估值
                    with torch.no_grad():
                        best_a     = policy_net(NS).argmax(1, keepdim=True)
                        next_q     = target_net(NS).gather(1, best_a)
                        target_q   = R + (GAMMA ** N_STEP) * next_q * (1 - D)

                    current_q = policy_net(S).gather(1, A)

                    td_errors = (current_q - target_q).detach().cpu().numpy().squeeze()
                    memory.update_priorities(idxs, td_errors)

                    loss = (W * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # 收緊裁剪，防止突發大梯度
                    optimizer.step()

                    # 軟更新 target net
                    soft_update(target_net, policy_net, TAU)

            print(
                f"回合: {episode+1:4d} | 總獎勵: {total_reward:7.2f} | "
                f"ε: {eps:.4f} | 步數: {steps_done} | 記憶: {len(memory)} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            # 自動存檔歷史最高分模型
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy_net.state_dict(), best_model_path)
                print(f"  ⭐ 新最高分 {best_reward:.2f}！模型存檔至 {best_model_path}")

            scheduler.step()  # 每回合更新 LR scheduler

            if (episode + 1) % SAVE_EVERY == 0:
                torch.save(policy_net.state_dict(), save_model_path)
                print(f"  → 模型已儲存至 {save_model_path}")

    except KeyboardInterrupt:
        print("\n[⚠️ 手動中斷訓練]")
        torch.save(policy_net.state_dict(), save_model_path)
    except WebDriverException:
        print("\n[⚠️ Chrome 視窗已關閉]")
        torch.save(policy_net.state_dict(), save_model_path)
    finally:
        env.close()


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_TO_LOAD = "dino10_ddqn_dueling_best.pth"                    # 填入 .pth 路徑可繼續訓練
    SAVE_PATH     = "dino11_ddqn_dueling.pth"
    train(load_model_path=MODEL_TO_LOAD, save_model_path=SAVE_PATH)

