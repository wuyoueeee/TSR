# 🚦 基于YOLO和大语言模型的交通标识智能识别系统 (Traffic Sign Recognition System)

> 基于 YOLO 和 LLaVA 大模型的智能交通标识检测与分析平台

>代码获取：[https://mbd.pub/o/bread/YZWamZ9pag==](https://mbd.pub/o/bread/YZWamZ9pag==)
## � 项目简介 (Introduction)

本项目是一个先进的交通标识识别系统，旨在提供高精度的交通标志检测和深度的场景理解能力。系统结合了 **YOLO** 的实时目标检测能力和 **LLaVA** (Large Language-and-Vision Assistant) 多模态大语言模型的语义理解能力。

通过友好的 Gradio Web 界面，用户可以轻松进行单张图片识别、批量数据处理以及实时摄像头检测。系统还内置了模型训练模块，通过界面即可启动模型的微调训练。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78ea57d09ed64d7d9824c0287670dedd.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/041c3c0202384766aba297c0e93ba178.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/416c85157ddf41eebe8f5312b82af2e0.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9f0d8949591743de8006763c140ad8e0.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6d78a11f50024d27a34466f36cf1b415.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a93dc95d5a07479ead2a1d964040cdbf.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4ed0d67da2dd4852a16890eb3e009bb5.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/526d7429d49c483291a6ae41d27a1605.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/565fad87a6e54655bd9a791a1671bdea.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d6c3461b5dc40ca918d8deb7420d185.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/50684bee72b749fd912bf9c194654ab1.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e334dfb052164c6c948a77c6868e57d4.jpeg#pic_center)

## ✨ 项目特点 (Features)

*   **🎯 高精度识别**: 采用 YOLO 算法，支持多种交通标识（限速、禁行、指示、警告等）的实时检测。
*   **🧠 深度智能分析**: 集成 LLaVA 多模态大模型，不仅能"看到"标识，还能理解标识的含义、位置及周边环境，生成中文分析报告。
*   **💻 全功能 Web 界面**: 基于 Gradio 构建的现代 UI，通过浏览器即可完成所有操作。
*   **🎞️ 多种检测模式**:
    *   **单图分析**: 详情扫描，支持生成 HTML 格式的专业检测报告。
    *   **批量处理**: 支持多文件上传，自动打包结果并生成统计 CSV。
    *   **实时监测**: 调用本地摄像头进行实时的交通标识捕捉。
*   **⚙️ 在线训练**: 内置训练画板，可调整 Epoch、Batch Size 等参数，一键启动模型微调。
*   **📊 历史记录管理**: 自动保存检测历史，支持数据导出和统计看板。

## �️ 安装方式 (Installation)

### 1. 环境准备
*   Python 3.8 或更高版本
*   CUDA 支持 (推荐用于 GPU 加速)
*   [Ollama](https://ollama.ai/) (如果需要使用 LLaVA 深度分析功能)

### 2. 克隆项目 & 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 LLaVA (可选)
如果您启用了深度分析功能，请确保本地安装并启动了 Ollama，并拉取了 `llava` 模型：
```bash
# 安装 Ollama 后运行
ollama serve
# 在另一个终端运行
ollama pull llava
```

## 🚀 使用方法 (Usage)

### 启动系统
在项目根目录下运行以下命令启动 Web 界面：
```bash
python run_web_advanced.py
```
启动后，浏览器访问 `http://localhost:7860` 即可使用。

### ⚙️ GPU 与 CPU 配置
无需修改代码，系统会自动检测 GPU。您也可以在 `config.yaml` 中强制指定设备：

**使用 GPU (默认推荐)**:
```yaml
yolov8:
  train:
    device: "0"  # 使用第一块 GPU
```

**使用 CPU (无显卡环境)**:
```yaml
yolov8:
  train:
    device: "cpu"
```

## 📂 目录结构 (Directory Structure)

```
TSR/
├── config.yaml                 # 系统总配置文件 (YOLO, LLaVA, Web)
├── requirements.txt            # 项目依赖列表
├── README.md                   # 项目说明文档
├── tsr_data.yaml               # YOLO 数据集配置文件
├── yolo11n.pt / yolov8n.pt     # 预训练权重文件
├── src/                        # 源代码目录
│   ├── web_interface_advanced.py  # Web 界面主程序
│   ├── yolo_detector.py           # YOLO 检测器核心封装
│   ├── traffic_sign_detection_system.py # 系统业务逻辑层
│   ├── history_manager.py         # 历史记录管理
│   └── convert_tsr_to_yolo.py     # 数据集转换工具
├── data/                       # 数据存放目录 (上传的图片/生成的报告)
├── runs/                       # 训练和检测结果输出目录
├── models/                     # 模型保存目录
└── logs/                       # 运行日志
```

## 🔧 配置说明 (Configuration)

项目的主要配置位于 `config.yaml` 文件中：

*   **yolov8**: 配置模型大小 (`n`, `s`, `m`, `l`, `x`)、训练参数 (epochs, batch_size) 和推理阈值。
*   **llava**: 配置 Ollama 的地址 (默认 `http://localhost:11434`)、模型名称和 Prompt 提示词模板。
*   **web**: 配置 Gradio 服务的端口和主机地址。
*   **system**: 配置输出和日志路径。

## 📊 数据集说明 (Dataset)

本项目使用 **TSRD (Chinese Traffic Sign Recognition Database)** 格式的数据集进行训练，并已转换为标准的 YOLO 格式。数据集共包含 **58** 个类别的中国交通标志。


### 📁 目录结构
```
data/
├── images/
│   ├── train/  # 训练集图片
│   └── val/    # 验证集图片
└── labels/
    ├── train/  # 训练集标签 (YOLO格式 txt)
    └── val/    # 验证集标签 (YOLO格式 txt)
```

### 🏷️ 类别详情 (Classes)
本系统支持识别以下 58 种交通标识：

<details>
<summary>👉 点击展开查看完整类别列表 (Click to expand)</summary>

| ID | 名称 (Name) | ID | 名称 (Name) |
|:---:|---|:---:|---|
| **0** | 限速 5km/h | **29** | 应当鸣笛 |
| **1** | 限速 15km/h | **30** | 非机动车行驶 |
| **2** | 限速 30km/h | **31** | 掉头 |
| **3** | 限速 40km/h | **32** | 路面双向通行 |
| **4** | 限速 50km/h | **33** | 注意信号灯标志 |
| **5** | 限速 60km/h | **34** | 注意危险 |
| **6** | 限速 70km/h | **35** | 注意行人 |
| **7** | 限速 80km/h | **36** | 注意非机动车 |
| **8** | 禁止左转或直行 | **37** | 注意儿童 |
| **9** | 禁止右转或直行 | **38** | 向右急转弯 |
| **10** | 禁止直行 | **39** | 向左急转弯 |
| **11** | 禁止左转 | **40** | 左侧变窄 |
| **12** | 禁止向左和向右转 | **41** | 下陡坡 |
| **13** | 禁止右转 | **42** | 慢行 |
| **14** | 禁止超车 | **43** | 左侧交叉路口 |
| **15** | 禁止掉头 | **44** | 十字交叉口 |
| **16** | 禁止机动车通行 | **45** | 村庄 |
| **17** | 禁止鸣笛 | **46** | 反向弯道 |
| **18** | 最低限速40 | **47** | 铁路道口 |
| **19** | 最低限速50 | **48** | 施工路段 |
| **20** | 直行或右转 | **49** | 连续弯道 |
| **21** | 直行 | **50** | 路面不平 |
| **22** | 左转 | **51** | 前方施工 |
| **23** | 左转或右转 | **52** | 减速让行 |
| **24** | 右转 | **53** | 停车让行 |
| **25** | 左侧行驶 | **54** | 封闭道路 |
| **26** | 右侧行驶 | **55** | 禁止通行 |
| **27** | 环形岛 | **56** | 注意让行 |
| **28** | 机动车行驶 | **57** | 检查 |

</details>

在 `tsr_data.yaml` 配置文件中定义了详细的路径和类别名称。

## 💻 核心代码 (Core Code)

### 检测器加载与推理 (`src/yolo_detector.py`)
```python
def predict(self, image_path: str, conf: float = 0.25) -> Tuple[list, np.ndarray]:
    # 加载模型
    if self.model is None:
        self.reload_model()
    
    # 执行推理
    results = self.model.predict(
        source=image_path,
        conf=conf,
        save=False
    )
    
    # 结果可视化绘制...
    return results, annotated_img
```

### Web 界面交互 (`src/web_interface_advanced.py`)
```python
def detect_image(self, image, conf, use_llava):
    # 保存临时文件
    # ...
    # 调用检测系统
    plot_img, detections, report, _, _ = self.system.detect_image(temp_path, conf=conf, use_llava=use_llava)
    
    # 生成统计 HTML
    if len(detections) > 0:
        stat = f"<div>Traffic Sign Detected: {len(detections)}</div>"
    
    return display_img, report, stat
```

### 🧠 自适应阈值检测 (`src/traffic_sign_detection_system.py`)
系统针对稀有类别（如施工、检查等）采用了自适应阈值策略，以提高召回率：
```python
# --- Adaptive Thresholding Logic ---
# 如果是稀有类别，接受较低的置信度 (>= 0.05)
# 如果是常见类别，执行严格的用户设定阈值 'conf'
is_rare = cls_id in self.RARE_CLASSES

if is_rare:
    if prob < 0.05: continue # Absolute minimum
else:
    if prob < conf: continue # Enforce standard threshold
```



### 💾 历史记录数据库管理 (`src/history_manager.py`)
使用 SQLite 自动记录每一次检测结果：
```python
def _init_database(self):
    conn = sqlite3.connect(str(self.db_path))
    cursor = conn.cursor()
    
    # 创建检测历史表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            task_type TEXT NOT NULL,
            source_path TEXT NOT NULL,
            num_defects INTEGER DEFAULT 0,
            use_llava BOOLEAN DEFAULT 0,
            result_path TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
```
