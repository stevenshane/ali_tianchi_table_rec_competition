"""全局配置：模型、BaseURL、API Key（从环境变量读取）、阈值与路径。"""
import os
from pathlib import Path

# ---------------- 模型与接口 ----------------
API_KEY_ENV = "DASHSCOPE_API_KEY"                      # API Key 所在的环境变量名
BASE_URL = "https://token-plan.maas.qianwenaiapi.com/apps/anthropic"

VISION_MODEL = "qwen3.8-max"     # 表格识别 / 版面分析
REASONING_MODEL = "qwen3.8-max"  # 题目解答
VISION_EFFORT = "high"
REASONING_EFFORT = "high"
LAYOUT_EFFORT = "low"
MAX_TOKENS = 983616

# 服务端拒答回退（仅 Claude API 直连时可用；千问等代理 BASE_URL 不支持，默认关闭）
ENABLE_FALLBACKS = os.getenv("TR_ENABLE_FALLBACKS", "0") == "1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# ---------------- 阈值 ----------------
CONF_THRESHOLD = 60            # 置信度低于此值触发二次放大识别
MIN_SHORT_EDGE = 1000          # 短边小于此值需放大
MIN_CHAR_H = 16                # 平均字符高度小于此值需放大
UPSCALE_FACTOR = 2
SKEW_DEG = 1.0                 # |skew| 大于此值做倾斜纠正
MAX_SKEW_DEG = 10.0            # 超过该角度不当作小角度倾斜处理
LOW_CONTRAST = 90              # 灰度 p95-p5 小于此值视为低对比度
UNEVEN_LIGHT = 25              # 分块平均亮度标准差大于此值视为光照不均
NOISE_TINY_FRAC = 0.6          # 孤立小连通域(<=4px)占全部连通域比例大于此值
NOISE_MIN_COUNT = 300          # 且数量大于此值，视为噪声明显
PERSPECTIVE_MIN_CONF = 0.85    # 四角置信度阈值
REFINE_ZOOM = 3                # 二次识别时关键区域放大倍数

PDF_DPI = 300
PDF_DPI_SMALL_FONT = 400
PDF_PAGE_BATCH = 20            # PDF 每批最多读取页数
PDF_TEXT_MIN_CHARS = 30        # 页面文字层字符数大于此值视为有文字层
PDF_TEXT_MIN_NORMAL_RATIO = 0.8  # 文字层中正常字符占比低于此值视为乱码（字体编码错乱）
MAX_VLM_EDGE = 2400            # 送入视觉模型的图片最长边，超过则切片
TILE_OVERLAP = 120

# ---------------- 路径 ----------------
ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
FILES_JSON_DIR = ROOT / "files_json"
WORK_DIR = ROOT / "work"
TESTS_XLSX = ROOT / "tests.xlsx"
SUBMISSION_DIR = ROOT          # 提交结果按 submission_yyyyMMdd_HHmmss.xlsx 命名
LOW_CONF_TXT = ROOT / "low_confidence.txt"
LOG_DIR = ROOT / "logs"

# 依赖安装镜像（仅作文档用途）
PIP_INDEX = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
