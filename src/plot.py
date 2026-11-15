# 그래프 기본 설정 (한글 폰트)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns

fonts = {f.name for f in fm.fontManager.ttflist}
for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if name in fonts:
        KFONT = name
        break
else:
    KFONT = "DejaVu Sans"

sns.set_theme(style="whitegrid", font=KFONT)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 200

PALETTE = sns.color_palette()   # seaborn 기본 색


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
