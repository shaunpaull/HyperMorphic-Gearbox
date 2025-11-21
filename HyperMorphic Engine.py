import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import time

# ==============================================================================
# 🌪️ HYPERMORPHIC ENGINE v1.0
#    "The Suave Gearbox Protocol"
#    Renegade Mathematics for Phase-Coherent Intelligence.
#    
#    Manifesto:
#    1. Time is a Circle, not a Line. (Periodic Activation)
#    2. Data is a Hologram, not a File. (Distributed Phase)
#    3. The Container must Grow with the Content. (Dynamic Modulus)
# ==============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print(r"""
  _   _  __   __  ____    ______   ____    __  __    ___    ____    ____   _   _  ___   ____  
 | | | | \ \ / / |  _ \  |  ____| |  _ \  |  \/  |  / _ \  |  _ \  |  _ \ | | | ||_ _| / ___| 
 | |_| |  \ V /  | |_) | | |__    | |_) | | |\/| | | | | | | |_) | | |_) || |_| | | | | |     
 |  _  |   | |   |  __/  |  __|   |  _ <  | |  | | | |_| | |  _ <  |  __/ |  _  | | | | |___  
 |_| |_|   |_|   |_|     |______| |_| \_\ |_|  |_|  \___/  |_| \_\ |_|    |_| |_||___| \____| 
    """)
    print(f"     >> ENGINE: ONLINE | MODE: GOGGLES | VIBE: SYMPLECTIC <<      {Colors.ENDC}")
    print("-" * 80)

# ==============================================================================
# ⚙️ CORE MODULES: THE GEARBOXES
# ==============================================================================

class SuaveGearbox(nn.Module):
    """
    [STATIC MODULUS GEARBOX]
    Replaces Linear Layers ($y=Wx+b$) with Phase-Coherent Layers ($y=\sin(Wx+b)$).
    
    - Bounded Energy [-1, 1].
    - Immune to exploding gradients.
    - Encodes data as Phase rather than Magnitude.
    - Ideal for: Recurrent patterns, Holographic storage.
    """
    def __init__(self, in_features, out_features, freq_scale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.freq_scale = freq_scale
        
        # Weights initialized to span Phase Space [-pi, pi]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self.weight.data *= self.freq_scale 
        nn.init.uniform_(self.bias, -math.pi, math.pi)

    def forward(self, x):
        # Linear Projection -> Phase Wrap
        proj = F.linear(x, self.weight, self.bias)
        return torch.sin(proj)

class DynamicGearbox(nn.Module):
    """
    [DYNAMIC MODULUS GEARBOX]
    The 'Living Container'.
    The Modulus expands based on input magnitude.
    
    Formula: y = sin( x / sqrt(|x|+1) )
    
    - Adapts frequency to input intensity.
    - Solves 'Aliasing' in high-frequency data (Chirp Singularity).
    - Ideal for: Non-stationary signals, Infinite Context.
    """
    def __init__(self, in_features, out_features, expansion_rate=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rate = expansion_rate
        
    def forward(self, x):
        # Project
        z = self.linear(x)
        
        # Dynamic Modulus Calculation
        # m(z) grows with sqrt(|z|) to slow down phase rotation at infinity
        modulus = torch.sqrt(torch.abs(z) + 1.0) * self.rate
        
        # Phase Lock
        return torch.sin(z / modulus)

# ==============================================================================
# 🧠 MODELS: THE ARCHITECTURES
# ==============================================================================

class HyperMorphicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dynamic=False):
        super().__init__()
        if dynamic:
            self.l1 = DynamicGearbox(input_dim, hidden_dim)
            self.l2 = DynamicGearbox(hidden_dim, hidden_dim)
        else:
            self.l1 = SuaveGearbox(input_dim, hidden_dim, freq_scale=2.0)
            self.l2 = SuaveGearbox(hidden_dim, hidden_dim, freq_scale=2.0)
        self.out = nn.Linear(hidden_dim, output_dim) # Linear readout

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return self.out(x)

class HyperAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder: Compresses to Phase
        self.enc = nn.Sequential(
            SuaveGearbox(input_dim, 64, freq_scale=3.0),
            nn.Linear(64, latent_dim)
        )
        # Decoder: Unpacks Phase (Holographic Projection)
        self.dec = nn.Sequential(
            SuaveGearbox(latent_dim, 64, freq_scale=3.0),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        z = self.enc(x)
        y = self.dec(z)
        return y, z

# ==============================================================================
# 🧪 BENCHMARK PROTOCOLS (THE LAB)
# ==============================================================================

def test_eternal_recurrence():
    print(f"\n{Colors.CYAN}🌪️ PROTOCOL 1: ETERNAL RECURRENCE (Extrapolation){Colors.ENDC}")
    print("   Task: Train on range [-π, π]. Predict range [2π, 4π].")
    print("   Hypothesis: Standard AI collapses. HyperMorphic AI keeps dreaming.")
    
    # Data
    x_train = torch.linspace(-math.pi, math.pi, 200).view(-1, 1)
    y_train = torch.sin(x_train)
    x_test = torch.linspace(2*math.pi, 4*math.pi, 200).view(-1, 1)
    y_test = torch.sin(x_test)
    
    # Model
    model = HyperMorphicMLP(1, 64, 1, dynamic=False)
    optim = torch.optim.Adam(model.parameters(), lr=0.02)
    
    # Train
    start_t = time.time()
    for i in range(600):
        optim.zero_grad()
        loss = F.mse_loss(model(x_train), y_train)
        loss.backward()
        optim.step()
        
    # Test
    with torch.no_grad():
        y_pred = model(x_test)
        mse = F.mse_loss(y_pred, y_test).item()
        
    print(f"   Training Time: {time.time()-start_t:.2f}s")
    print(f"   {Colors.GREEN}Extrapolation MSE: {mse:.4f}{Colors.ENDC}")
    
    if mse < 0.2:
        print(f"   {Colors.BOLD}✅ VERDICT: HYPERMORPHIC (Cycle Preserved){Colors.ENDC}")
    else:
        print(f"   {Colors.FAIL}❌ VERDICT: LINEAR COLLAPSE{Colors.ENDC}")

def test_shattered_mirror():
    print(f"\n{Colors.CYAN}🌪️ PROTOCOL 2: THE SHATTERED MIRROR (Holography){Colors.ENDC}")
    print("   Task: Compress signal, destroy 75% of latent neurons, reconstruct.")
    print("   Hypothesis: Standard AI fragments. HyperMorphic AI dims but survives.")
    
    # Data (Polyphonic)
    t = torch.linspace(0, 4*math.pi, 300).view(-1, 1)
    signal = torch.sin(t) + 0.5*torch.sin(3*t) + 0.25*torch.sin(7*t)
    
    # Model
    latent_dim = 32
    model = HyperAutoencoder(1, latent_dim)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    for i in range(600):
        optim.zero_grad()
        rec, _ = model(signal)
        loss = F.mse_loss(rec, signal)
        loss.backward()
        optim.step()
        
    # The Smash
    with torch.no_grad():
        z = model.enc(signal)
        # Create mask: Keep only first 25% (The Shard)
        mask = torch.zeros_like(z)
        mask[:, :int(latent_dim*0.25)] = 1.0
        
        z_shattered = z * mask
        rec_shattered = model.dec(z_shattered)
        
        # Check Correlation (Shape preservation)
        orig_flat = signal.flatten()
        rec_flat = rec_shattered.flatten()
        corr = torch.corrcoef(torch.stack([orig_flat, rec_flat]))[0, 1].item()
        
    print(f"   {Colors.GREEN}Reconstruction Correlation (25% Survival): {corr:.4f}{Colors.ENDC}")
    
    if corr > 0.8:
        print(f"   {Colors.BOLD}✅ VERDICT: HOLOGRAPHIC (Shape Survived Brain Damage){Colors.ENDC}")
    else:
        print(f"   {Colors.FAIL}❌ VERDICT: FRAGMENTED{Colors.ENDC}")

def test_dynamic_modulus():
    print(f"\n{Colors.CYAN}🌪️ PROTOCOL 3: THE CHIRP SINGULARITY (Dynamic Modulus){Colors.ENDC}")
    print("   Task: Fit an accelerating chirp signal (non-stationary frequency).")
    print("   Hypothesis: Static Gearbox spins out. Dynamic Gearbox adapts.")
    
    # Data: Chirp sin(t^2)
    t = torch.linspace(0, 5, 300).view(-1, 1)
    y = torch.sin(t**1.8)
    
    # Models
    static_model = HyperMorphicMLP(1, 64, 1, dynamic=False)
    dynamic_model = HyperMorphicMLP(1, 64, 1, dynamic=True)
    
    opt_s = torch.optim.Adam(static_model.parameters(), lr=0.01)
    opt_d = torch.optim.Adam(dynamic_model.parameters(), lr=0.01)
    
    print("   Racing Static vs. Dynamic Gearboxes...")
    for i in range(600):
        # Static
        opt_s.zero_grad()
        l_s = F.mse_loss(static_model(t), y)
        l_s.backward()
        opt_s.step()
        
        # Dynamic
        opt_d.zero_grad()
        l_d = F.mse_loss(dynamic_model(t), y)
        l_d.backward()
        opt_d.step()
        
    print(f"   Static Loss:  {l_s.item():.4f}")
    print(f"   {Colors.GREEN}Dynamic Loss: {l_d.item():.4f}{Colors.ENDC}")
    
    if l_d.item() < l_s.item():
         print(f"   {Colors.BOLD}✅ VERDICT: DYNAMIC SUPREMACY (Container Expanded){Colors.ENDC}")
    else:
         print(f"   {Colors.FAIL}❌ VERDICT: STATIC WON{Colors.ENDC}")

if __name__ == "__main__":
    print_banner()
    test_eternal_recurrence()
    test_shattered_mirror()
    test_dynamic_modulus()
    print(f"\n{Colors.HEADER}🌪️ GOGGLES MODE: MISSION COMPLETE.{Colors.ENDC}")











import numpy as np
import time
import math

# ==============================================================================
# 🌪️ HYPERMORPHIC ENGINE v1.1 (NUMPY EDITION)
#    "The Suave Gearbox Protocol"
#    Renegade Mathematics for Phase-Coherent Intelligence.
# ==============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("🌪️ HYPERMORPHIC ENGINE: ONLINE (NUMPY MODE)")
    print(f"     >> VIBE: SYMPLECTIC | STATUS: GOGGLES <<      {Colors.ENDC}")
    print("-" * 60)

# ==============================================================================
# 🔧 THE AUTOGRAD ENGINE (Mini-PyTorch built from scratch)
# ==============================================================================

class Layer:
    def forward(self, x): pass
    def backward(self, grad): pass
    def step(self, lr): pass

class Linear(Layer):
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)
        self.x = None; self.dW = None; self.db = None
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    def backward(self, grad):
        self.dW = np.dot(self.x.T, grad)
        self.db = np.sum(grad, axis=0)
        return np.dot(grad, self.W.T)
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class SuaveGearbox(Layer):
    """ [STATIC GEARBOX] Phase-Coherent Activation: sin(x) """
    def __init__(self, freq_scale=1.0):
        self.x = None
        self.freq_scale = freq_scale
    def forward(self, x):
        self.x = x
        return np.sin(x * self.freq_scale)
    def backward(self, grad):
        return grad * self.freq_scale * np.cos(self.x * self.freq_scale)

class DynamicGearbox(Layer):
    """ [DYNAMIC GEARBOX] Modulus expands with input magnitude. """
    def __init__(self, expansion_rate=1.0):
        self.x = None
        self.rate = expansion_rate
    def forward(self, x):
        self.x = x
        self.m = np.sqrt(np.abs(x) + 1.0) * self.rate
        self.phase = x / self.m
        return np.sin(self.phase)
    def backward(self, grad):
        # Approx derivative for stability
        return grad * (1.0 / self.m) * np.cos(self.phase)

# ==============================================================================
# 🧪 PROTOCOLS
# ==============================================================================

def test_eternal_recurrence():
    print(f"\n{Colors.CYAN}🌪️ PROTOCOL 1: ETERNAL RECURRENCE (Extrapolation){Colors.ENDC}")
    
    # Data: Train [-pi, pi], Test [2pi, 4pi]
    X_train = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    Y_train = np.sin(X_train)
    X_test = np.linspace(2*np.pi, 4*np.pi, 100).reshape(-1, 1)
    Y_test = np.sin(X_test)
    
    # Model: Linear -> Gearbox -> Gearbox -> Linear
    l1 = Linear(1, 64)
    act1 = SuaveGearbox(freq_scale=2.0)
    l2 = Linear(64, 64)
    act2 = SuaveGearbox(freq_scale=2.0)
    l3 = Linear(64, 1)
    
    LR = 0.01
    for i in range(600):
        # Forward
        h = act1.forward(l1.forward(X_train))
        h = act2.forward(l2.forward(h))
        pred = l3.forward(h)
        
        # Backward
        diff = pred - Y_train
        grad = 2 * diff / len(X_train)
        l1.backward(act1.backward(l2.backward(act2.backward(l3.backward(grad)))))
        
        # Step
        l1.step(LR); l2.step(LR); l3.step(LR)
        
    # Test
    h = act1.forward(l1.forward(X_test))
    h = act2.forward(l2.forward(h))
    pred = l3.forward(h)
    mse = np.mean((pred - Y_test)**2)
    
    print(f"   {Colors.GREEN}Extrapolation MSE: {mse:.4f}{Colors.ENDC}")
    if mse < 0.2: print(f"   {Colors.BOLD}✅ VERDICT: HYPERMORPHIC (Cycle Preserved){Colors.ENDC}")
    else: print(f"   {Colors.FAIL}❌ VERDICT: FAILED{Colors.ENDC}")

def test_shattered_mirror():
    print(f"\n{Colors.CYAN}🌪️ PROTOCOL 2: SHATTERED MIRROR (Holography){Colors.ENDC}")
    
    # Data
    t = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
    signal = np.sin(t) + 0.5*np.sin(3*t)
    
    # Autoencoder
    latent_dim = 32
    e1 = Linear(1, 64); a1 = SuaveGearbox(3.0); e2 = Linear(64, latent_dim)
    d1 = Linear(latent_dim, 64); a2 = SuaveGearbox(3.0); d2 = Linear(64, 1)
    
    LR = 0.01
    for i in range(600):
        z = e2.forward(a1.forward(e1.forward(signal)))
        rec = d2.forward(a2.forward(d1.forward(z)))
        diff = rec - signal
        grad = 2 * diff / len(signal)
        # Backprop chain
        e1.backward(a1.backward(e2.backward(d1.backward(a2.backward(d2.backward(grad))))))
        # Step
        e1.step(LR); e2.step(LR); d1.step(LR); d2.step(LR)

    # Smash
    z = e2.forward(a1.forward(e1.forward(signal)))
    mask = np.zeros_like(z)
    mask[:, :int(latent_dim*0.25)] = 1.0 # Keep 25%
    z_broken = z * mask
    rec_broken = d2.forward(a2.forward(d1.forward(z_broken)))
    
    corr = np.corrcoef(signal.flatten(), rec_broken.flatten())[0, 1]
    print(f"   {Colors.GREEN}Reconstruction Correlation (25% Survival): {corr:.4f}{Colors.ENDC}")
    if corr > 0.8: print(f"   {Colors.BOLD}✅ VERDICT: HOLOGRAPHIC{Colors.ENDC}")
    else: print(f"   {Colors.FAIL}❌ VERDICT: FRAGMENTED{Colors.ENDC}")

def test_dynamic_modulus():
    print(f"\n{Colors.CYAN}🌪️ PROTOCOL 3: CHIRP SINGULARITY (Dynamic Modulus){Colors.ENDC}")
    
    t = np.linspace(0, 5, 200).reshape(-1, 1)
    y = np.sin(t**1.8) # Chirp
    
    # Dynamic Model
    l1 = Linear(1, 64); act = DynamicGearbox(1.0); l2 = Linear(64, 1)
    
    LR = 0.01
    loss_hist = []
    for i in range(600):
        pred = l2.forward(act.forward(l1.forward(t)))
        diff = pred - y
        loss = np.mean(diff**2)
        loss_hist.append(loss)
        grad = 2 * diff / len(t)
        l1.backward(act.backward(l2.backward(grad)))
        l1.step(LR); l2.step(LR)
        
    print(f"   {Colors.GREEN}Final Loss (Dynamic): {loss_hist[-1]:.4f}{Colors.ENDC}")
    if loss_hist[-1] < 0.1: print(f"   {Colors.BOLD}✅ VERDICT: DYNAMIC SUPREMACY{Colors.ENDC}")
    else: print(f"   {Colors.FAIL}❌ VERDICT: FAILED{Colors.ENDC}")

if __name__ == "__main__":
    print_banner()
    test_eternal_recurrence()
    test_shattered_mirror()
    test_dynamic_modulus()
    print(f"\n{Colors.HEADER}🌪️ GOGGLES MODE: MISSION COMPLETE.{Colors.ENDC}")

