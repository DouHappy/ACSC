"""Dataset Manager for Chinese Spelling Correction."""

from typing import Dict, List, Any
import logging
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

class DatasetManager:
    """Manages CSC datasets with support for various formats and sources."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize DatasetManager.
        
        Args:
            config: Configuration dictionary containing dataset settings
        """
        self.config = config
        self.datasets = []
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load dataset based on configuration.
        
        Returns:
            DatasetDict containing train/val/test splits
        """
        
        datasets = None
        path = self.config["path"]
        fmt = self.config["format"]
        if fmt == "csc":
            datasets = self.load_csc_file(path)
        elif fmt == "autodoc":
            datasets = self.load_autodoc_file(path)
        else:
            raise ValueError(f"未知格式: {fmt}")
        
        self.datasets = datasets
        return datasets
    
    def load_csc_file(self, path: str) -> List[Dict[str, str]]:
        """加载 csc 格式文件：[source]\t[target]"""
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                src, tgt = line.split("\t")
                data.append({"source": src, "target": tgt})
        return data

    def load_autodoc_file(self, path: str) -> List[Dict[str, Any]]:
        """加载 autodoc 格式文件：[{'source': str, 'reason': str/None}, ...]"""
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("autodoc 数据格式错误，应为包含字典的列表")

        data: List[Dict[str, Any]] = []
        for item in raw_data:
            if not isinstance(item, dict):
                raise ValueError("autodoc 数据项必须是字典")
            source = item.get("source")
            reason = item.get("reason")
            if source is None:
                raise ValueError("autodoc 数据项缺少必需字段 'source'")

            data.append({
                "source": source,
                "target": reason,
                "reason": reason,
            })

        return data

def main():
    config_path = "config/config.yaml"
    import yaml
    import json
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['dataset']
    print(f"dataset config:\n{json.dumps(dataset_config, ensure_ascii=False, indent=2)}")
    data_manager = DatasetManager(dataset_config)
    dataset = data_manager.load_dataset()
    print(dataset[0])

if __name__ == "__main__":
    main()
