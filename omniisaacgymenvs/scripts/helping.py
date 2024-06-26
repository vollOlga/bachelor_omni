import torch

# Pfad zur .pth Datei
checkpoint_path = '/home/vncusr/OmniIsaacGymEnvs/omniisaacgymenvs/runs/UR10Reacher/nn/last_UR10Reacher_ep_251200_rew_1971.0665.pth'

# Lädt den Checkpoint
checkpoint = torch.load(checkpoint_path)

# Zeigt alle Schlüssel im Checkpoint Dictionary an
print("Keys in checkpoint:", checkpoint.keys())

# Zeigt die Struktur der einzelnen Elemente an
print("Actor Model Parameters:")
for name, param in checkpoint['actor'].items():
    print(f"{name}: {param.size()}")

# Details für Critic Modell
print("\nCritic Model Parameters:")
for name, param in checkpoint['critic'].items():
    print(f"{name}: {param.size()}")

# Details für Critic Target Modell
print("\nCritic Target Model Parameters:")
for name, param in checkpoint['critic_target'].items():
    print(f"{name}: {param.size()}")
