# Rex-Omni API

## 最新更新 ✨

已添加**中文标签支持**和**更细的检测框线条**！

### 改进内容

1. **支持中文标签** - 可视化结果中的标签现在可以正确显示中文
2. **更细的检测框** - 线条宽度从 5 降低到 2，更加美观

## 安装中文字体

### Ubuntu/Debian 系统

```bash
sudo apt-get update
sudo apt-get install fonts-wqy-zenhei
```

字体路径：`/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc`

### CentOS/RedHat 系统

```bash
sudo yum install wqy-zenhei-fonts
```

字体路径：`/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc`

### macOS 系统

macOS 系统自带中文字体，使用：

```bash
python api_server.py --font_path /System/Library/Fonts/PingFang.ttc
```

### Windows 系统

Windows 系统自带中文字体，使用：

```bash
python api_server.py --font_path C:/Windows/Fonts/simhei.ttf
```

## 启动服务器

### 基本启动（使用默认字体）

```bash
python api/api_server.py
```

### 指定自定义字体路径

```bash
python api/api_server.py --font_path /path/to/your/font.ttf
```

### 完整示例

```bash
python api/api_server.py \
    --model_path IDEA-Research/Rex-Omni \
    --backend transformers \
    --host 0.0.0.0 \
    --port 8000 \
    --font_path /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
```

## API 使用示例

### 检测中文类别

```python
import requests

# 上传图像
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    data = {
        'request_json': json.dumps({
            'task': 'detection',
            'categories': ['人', '狗', '汽车'],  # 中文类别
            'return_visualization': True
        })
    }
    response = requests.post('http://localhost:8000/api/detect', files=files, data=data)

result = response.json()
# result['visualization'] 包含带中文标签的可视化图像（base64 编码）
```

### 使用简化接口

```python
import requests

# 直接使用 detect_for_chat 接口
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/detect_for_chat',
        files={'image': f},
        data={'categories': '人, 狗, 汽车'}  # 中文类别
    )

result = response.json()
print(result['detection_results'])
```

## 故障排除

### 1. 中文显示为方框

**原因**：系统未安装中文字体或字体路径错误

**解决方案**：
```bash
# 1. 安装中文字体
sudo apt-get install fonts-wqy-zenhei

# 2. 检查字体文件是否存在
ls -la /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc

# 3. 使用正确的字体路径启动
python api/api_server.py --font_path /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
```

### 2. 字体路径找不到

**查找系统中的中文字体**：
```bash
# Linux 系统
fc-list :lang=zh

# 或者
find /usr/share/fonts -name "*.ttf" -o -name "*.ttc" | grep -i zh
```

### 3. 检测框太粗或太细

如果需要调整线条粗细，编辑 `api/api_server.py` 文件：

```python
# 找到所有 RexOmniVisualize 调用，修改 draw_width 参数
vis_image = RexOmniVisualize(
    image=img,
    predictions=predictions,
    font_size=20,
    draw_width=2,  # 修改这里：1=很细，3=中等，5=粗
    show_labels=True,
    font_path=CHINESE_FONT_PATH
)
```

## 配置参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--font_path` | `/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc` | 中文字体路径 |
| `draw_width` | `2` | 检测框线条宽度（代码中） |
| `font_size` | `20` | 标签字体大小（代码中） |

## 相关文件

- `api/api_server.py` - API 服务器主文件（已修改）
- `rex_omni/utils.py` - 可视化工具（支持 font_path 参数）

## 技术细节

修改的主要位置：

1. **全局配置**（第 62-68 行）：添加 `CHINESE_FONT_PATH` 变量
2. **命令行参数**（第 660-661 行）：添加 `--font_path` 参数
3. **所有可视化调用**：更新 `draw_width=2` 和 `font_path=CHINESE_FONT_PATH`
4. **主函数**（第 676-686 行）：设置全局字体路径

## 测试

```bash
# 启动服务器
python api/api_server.py

# 测试健康检查
curl http://localhost:8000/health

# 测试中文检测（需要准备测试图片）
curl -X POST "http://localhost:8000/api/detect_for_chat" \
  -F "image=@test.jpg" \
  -F "categories=人, 狗, 猫"
```
