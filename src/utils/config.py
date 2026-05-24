"""
配置管理
========
统一管理系统配置
"""

from pathlib import Path
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"

# 默认配置
DEFAULT_CONFIG = {
    "paths": {
        "data_dir": "data",
        "indices_dir": "indices",
        "vector_store": "indices/vector_store",
        "graph_store": "indices/graph.pkl",
    },
    "retrieval": {
        "vector_k": 10,
        "top_n": 5,
        "rrf_k_const": 60,
    },
    "classifier": {
        "delivery_keywords": [
            "外卖", "饿了么", "美团", "送餐", "取餐", "骑手", "小哥",
            "快递", "包裹", "顺丰", "中通", "圆通", "韵达", "菜鸟"
        ],
        "risk_keywords": [
            "转账", "汇款", "验证码", "安全账户", "账户异常",
            "涉嫌", "案件", "中奖", "恭喜", "免费领取"
        ],
    },
    "generation": {
        "temperature": 0.3,
        "max_tokens": 512,
    }
}


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，默认使用 configs/config.yaml

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = CONFIG_FILE

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
            # 合并配置
            config = DEFAULT_CONFIG.copy()
            if user_config:
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value
            return config

    return DEFAULT_CONFIG


def get_path(key: str, config: dict = None) -> Path:
    """获取路径配置"""
    if config is None:
        config = load_config()

    path_str = config.get("paths", {}).get(key, key)
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # 确保目录存在
    if key.endswith("_dir") or key == "vector_store":
        path.mkdir(parents=True, exist_ok=True)

    return path
