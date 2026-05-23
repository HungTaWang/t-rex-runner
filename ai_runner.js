/**
 * AI Runner for Chrome T-Rex Game using ONNX Runtime Web
 */

let aiEnabled = false;
let session = null;
let aiLoopId = null;
let modelLoaded = false;
let btn = null;

// The state extraction logic translated from Python to JS
function getGameState() {
    if (!Runner || !Runner.instance_) return null;
    const r = Runner.instance_;
    
    // If not playing or crashed, don't return state
    if (!r.playing || r.crashed) return null;

    const obs = r.horizon.obstacles;
    const tx = r.tRex.xPos;
    const ty = r.tRex.yPos;
    const spd = r.currentSpeed || 6;
    
    // Exactly as the python _JS_GET_STATE
    const out = [spd / 13.0, ty / 150.0];
    for (let i = 0; i < 2; i++) {
        if (i < obs.length) {
            const o = obs[i];
            out.push((o.xPos - tx) / 600.0);
            out.push(o.yPos / 150.0);
            out.push(o.width / 100.0);
            out.push((o.typeConfig ? o.typeConfig.height : 50) / 100.0);
        } else {
            out.push(1.0, 0.0, 0.0, 0.5);
        }
    }
    return out;
}

function triggerJump() {
    // Release down key (just in case), press up key
    document.dispatchEvent(new KeyboardEvent('keyup', { 'keyCode': 40, 'which': 40 }));
    document.dispatchEvent(new KeyboardEvent('keydown', { 'keyCode': 38, 'which': 38 }));
}

function setupUI() {
    btn = document.createElement('button');
    btn.innerText = "Loading AI Model...";
    btn.disabled = true;
    
    // Default to centered and prominent styling
    btn.style.position = "absolute";
    btn.style.top = "50%";
    btn.style.left = "50%";
    btn.style.transform = "translate(-50%, -50%)";
    btn.style.zIndex = "9999";
    btn.style.padding = "20px 40px";
    btn.style.fontSize = "24px";
    btn.style.fontWeight = "bold";
    btn.style.fontFamily = "monospace";
    btn.style.cursor = "not-allowed";
    btn.style.backgroundColor = "#535353";
    btn.style.color = "white";
    btn.style.border = "none";
    btn.style.borderRadius = "8px";
    btn.style.boxShadow = "0 4px 6px rgba(0,0,0,0.3)";
    btn.style.transition = "all 0.3s ease"; // Smooth transition

    btn.onclick = () => {
        if (!modelLoaded) return;
        aiEnabled = !aiEnabled;
        btn.innerText = aiEnabled ? "Disable AI" : "Enable AI";
        btn.style.backgroundColor = aiEnabled ? "#4CAF50" : "#535353";
        
        if (aiEnabled) {
            // Move to top-left corner
            btn.style.top = "20px";
            btn.style.left = "20px";
            btn.style.transform = "none";
            btn.style.padding = "10px 20px";
            btn.style.fontSize = "16px";
            
            // Automatically start the game if it hasn't started yet
            if (Runner && Runner.instance_ && !Runner.instance_.playing) {
                document.dispatchEvent(new KeyboardEvent('keydown',{'keyCode':32,'which':32}));
            }
            aiLoop();
        } else {
            // Move back to center
            btn.style.top = "50%";
            btn.style.left = "50%";
            btn.style.transform = "translate(-50%, -50%)";
            btn.style.padding = "20px 40px";
            btn.style.fontSize = "24px";

            if (aiLoopId) {
                cancelAnimationFrame(aiLoopId);
                aiLoopId = null;
            }
        }
    };

    document.body.appendChild(btn);
}

async function initONNX() {
    try {
        console.log("Loading ONNX model...");
        // Expects dino.onnx in the same directory
        session = await ort.InferenceSession.create('./dino.onnx', { executionProviders: ['wasm'] });
        console.log("ONNX model loaded successfully.");
        modelLoaded = true;
        
        // Update UI
        if (btn) {
            btn.innerText = "Enable AI";
            btn.disabled = false;
            btn.style.cursor = "pointer";
        }
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
        if (btn) {
            btn.innerText = "AI Load Failed";
            btn.style.backgroundColor = "red";
        }
    }
}

async function aiLoop() {
    if (!aiEnabled) return;

    const state = getGameState();
    if (state && session) {
        try {
            // Prepare the input tensor. The python model expects shape [1, 10]
            const inputTensor = new ort.Tensor('float32', Float32Array.from(state), [1, 10]);
            const feeds = { state: inputTensor };

            // Run inference
            const results = await session.run(feeds);
            const qValues = results.q_values.data; // Float32Array

            // Argmax
            let action = 0;
            if (qValues[1] > qValues[0]) {
                action = 1;
            }

            // Execute action
            if (action === 1) {
                triggerJump();
            }
        } catch (e) {
            console.error("Inference error:", e);
        }
    }

    // Schedule next loop
    aiLoopId = requestAnimationFrame(aiLoop);
}

// Start initialization when window loads
window.addEventListener('load', () => {
    // Setup UI immediately so user sees something is happening
    setupUI();
    // Wait a brief moment to ensure ort is fully parsed before loading model
    setTimeout(initONNX, 500);
});
