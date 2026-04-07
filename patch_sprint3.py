import re

file_path = 'docs/superpowers/plans/2026-04-07-sprint3-transformer-bc.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace return self.policy_head(x), self.value_head(x)
text = text.replace(
'''        return self.policy_head(x), self.value_head(x)''',
'''        # [FIX GRADIENT EARTHQUAKE] Cắt đứt gradient từ Value Head về phần thân Transformer
        # Ở PPO Phase 2, Value Head ngẫu nhiên => Error siêu lớn. Nếu k detach, nó phá nát Core đã train ở BC.
        return self.policy_head(x), self.value_head(x.detach())''')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Sprint 3 fully patched")
