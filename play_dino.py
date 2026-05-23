import os, time, threading
import http.server, socketserver
import numpy as np
import torch
import torch.nn as nn
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"運算設備: {device}")

# ─────────────────────────────────────────────────────────────────
# 路徑處理 (支援 PyInstaller .exe)
# ─────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 本地伺服器與 JS 參數
GAME_DIR = os.path.join(BASE_DIR, "t-rex-runner")
GAME_PORT = 8000

def _start_local_server():
    if not os.path.isdir(GAME_DIR):
        print(f"[警告] 找不到 {GAME_DIR}，改用線上版")
        return False
    import functools
    class SilentHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
    Handler = functools.partial(SilentHandler, directory=GAME_DIR)
    class QuietTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
        def handle_error(self, request, client_address):
            import sys
            exc = sys.exc_info()[1]
            if isinstance(exc, ConnectionResetError): return
            super().handle_error(request, client_address)
    try:
        httpd = QuietTCPServer(("", GAME_PORT), Handler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"[伺服器] 本地端運行於 http://localhost:{GAME_PORT}")
        return True
    except OSError:
        return True

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
            out.push(1.0, 0.0, 0.0, 0.5);
        }
    }
    return out;
})();
"""

STATE_DIM = 10

# ─────────────────────────────────────────────────────────────────
# 環境與網路結構 (需與訓練時完全相同)
# ─────────────────────────────────────────────────────────────────
class DinoEnv:
    def __init__(self, use_local=True):
        opts = Options()
        opts.add_argument("--mute-audio")
        opts.add_argument("--disable-web-security")
        self.driver = webdriver.Chrome(options=opts)
        url = f"http://localhost:{GAME_PORT}/index.html" if use_local else "https://chromedino.com/"
        self.driver.get(url)
        time.sleep(2)
        try:
            self.driver.execute_script("Runner.instance_.playIntro()")
        except:
            self.driver.execute_script("document.dispatchEvent(new KeyboardEvent('keydown',{'keyCode':32,'which':32}));")
        time.sleep(0.8)

    def _get_state(self):
        raw = self.driver.execute_script(_JS_GET_STATE)
        return np.array(raw, dtype=np.float32)

    def reset(self):
        self.driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.15)
        return self._get_state()

    _JS_UP_DOWN = "document.dispatchEvent(new KeyboardEvent('keydown',{'keyCode':38,'which':38}));"
    _JS_DN_UP   = "document.dispatchEvent(new KeyboardEvent('keyup', {'keyCode':40,'which':40}));"

    def step(self, action):
        if action == 1:
            self.driver.execute_script(self._JS_DN_UP)
            self.driver.execute_script(self._JS_UP_DOWN)
        time.sleep(0.05)
        crashed = self.driver.execute_script("return Runner.instance_.crashed")
        distance = self.driver.execute_script("return Runner.instance_.distanceRan")
        return self._get_state(), distance, bool(crashed)

    def close(self):
        try: self.driver.quit()
        except: pass

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    def forward(self, x):
        f = self.shared(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))

# ─────────────────────────────────────────────────────────────────
# 遊玩主程式
# ─────────────────────────────────────────────────────────────────
def play(model_path):
    if not os.path.exists(model_path):
        print(f"找不到模型檔案: {model_path}")
        print("請確保檔案名稱正確，且位於同一資料夾。")
        return
        
    use_local = _start_local_server()
    env = DinoEnv(use_local)
    
    net = DuelingDQN(STATE_DIM, 2).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print(f"成功載入模型: {model_path}")
    print("開始遊玩！(按 Ctrl+C 可停止)")

    try:
        episode = 1
        while True:
            state = env.reset()
            done = False
            score = 0
            while not done:
                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0).to(device)
                    # 100% 貪婪策略：永遠選擇 Q 值最高的動作
                    action = net(st).argmax(1).item()
                state, distance, done = env.step(action)
                score = max(score, distance * 0.025) # 轉換為遊戲內實際分數比例
            print(f"第 {episode:3d} 回合結束，遊戲分數: {int(score)}")
            episode += 1
            time.sleep(1) # 暫停一下再重來
    except KeyboardInterrupt:
        print("\n手動停止遊玩。")
    except WebDriverException:
        print("\n[Chrome 視窗已關閉]")
    finally:
        env.close()

if __name__ == "__main__":
    # 指定你的最佳模型檔案 (同樣使用 BASE_DIR 確保 exe 能找到)
    MODEL_PATH = os.path.join(BASE_DIR, "dino_ddqn_dueling_best.pth")
    play(MODEL_PATH)
