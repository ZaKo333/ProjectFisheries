"""验证项目本地 fastembed 缓存目录能否离线加载 bge-small-zh-v1.5。"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from fastembed import TextEmbedding  # noqa: E402

CACHE_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "fastembed"

t0 = time.perf_counter()
model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5", cache_dir=str(CACHE_DIR))
print("load seconds: %.2f" % (time.perf_counter() - t0))

docs = [
    "溶解氧低于 3 mg/L 时鱼会浮头，需要立即开启增氧机。",
    "水霉病由水霉菌引起，流行水温 10 到 20 摄氏度。",
    "投喂量应根据水温和溶氧调整，高温期应减少投喂。",
]
query = "池塘溶氧太低了怎么办"

vecs = list(model.embed(docs))
qvec = list(model.embed([query]))[0]

print("dim:", len(qvec))
import numpy as np  # noqa: E402

m = np.array(vecs)
sims = m @ np.array(qvec) / (np.linalg.norm(m, axis=1) * np.linalg.norm(qvec))
for d, s in zip(docs, sims):
    print("  %.4f  %s" % (s, d[:24]))
print("best:", docs[int(np.argmax(sims))][:20])
