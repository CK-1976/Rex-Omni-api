#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rex-Omni API 服务器
提供 RESTful API 接口用于目标检测
"""

import argparse
import base64
import io
import json
import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from rex_omni import RexOmniVisualize, RexOmniWrapper


# ==================== 数据模型定义 ====================

class DetectionRequest(BaseModel):
    """目标检测请求"""
    task: str = Field(
        default="detection",
        description="任务类型: detection, pointing, visual_prompting, keypoint, ocr_box, ocr_polygon"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="要检测的类别列表，例如: ['person', 'car', 'dog']"
    )
    keypoint_type: Optional[str] = Field(
        default=None,
        description="关键点类型（仅用于 keypoint 任务）: person, hand, animal"
    )
    visual_prompt_boxes: Optional[List[List[float]]] = Field(
        default=None,
        description="视觉提示框（仅用于 visual_prompting 任务）: [[x0, y0, x1, y1], ...]"
    )
    return_visualization: bool = Field(
        default=False,
        description="是否返回可视化图像（base64 编码）"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    backend: str
    supported_tasks: List[str]


# ==================== 全局变量 ====================

class AppConfig:
    """应用配置类"""
    def __init__(self):
        # 中文字体路径配置（用于可视化中文标签）
        # 常见的中文字体路径：
        # - Ubuntu/Debian: /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
        # - CentOS/RedHat: /usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc
        # - Windows: C:/Windows/Fonts/simhei.ttf
        # - macOS: /System/Library/Fonts/PingFang.ttc
        self.font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

# 全局配置实例
config = AppConfig()

app = FastAPI(
    title="Rex-Omni API",
    description="目标检测 API 服务",
    version="1.0.0"
)

# 配置 CORS（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
rex_model: Optional[RexOmniWrapper] = None


# ==================== 辅助函数 ====================

def load_image_from_upload(file: UploadFile) -> Image.Image:
    """从上传的文件加载图像"""
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法加载图像: {str(e)}")


def load_image_from_base64(base64_str: str) -> Image.Image:
    """从 base64 字符串加载图像"""
    try:
        # 移除可能的 data:image 前缀
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]

        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析 base64 图像: {str(e)}")


def image_to_base64(image: Image.Image) -> str:
    """将 PIL 图像转换为 base64 字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


# ==================== API 端点 ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 健康检查"""
    return {
        "status": "running",
        "model_loaded": rex_model is not None,
        "backend": rex_model.backend if rex_model else "unknown",
        "supported_tasks": rex_model.get_supported_tasks() if rex_model else []
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy" if rex_model is not None else "model_not_loaded",
        "model_loaded": rex_model is not None,
        "backend": rex_model.backend if rex_model else "unknown",
        "supported_tasks": rex_model.get_supported_tasks() if rex_model else []
    }


@app.post("/api/detect")
async def detect_objects(
    image: UploadFile = File(..., description="要检测的图像文件"),
    request_json: str = Form(..., description="JSON 格式的检测参数")
):
    """
    通用目标检测端点（对应 gradio_demo.py 的 run_inference 函数）

    接收图像文件和检测参数，返回检测结果
    支持所有任务类型：detection, keypoint, ocr_box, ocr_polygon, pointing, visual_prompting
    """
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # 解析请求参数
        request_data = json.loads(request_json)
        request = DetectionRequest(**request_data)

        # 加载图像
        img = load_image_from_upload(image)

        # 执行检测
        start_time = time.time()
        results = rex_model.inference(
            images=img,
            task=request.task,
            categories=request.categories,
            keypoint_type=request.keypoint_type,
            visual_prompt_boxes=request.visual_prompt_boxes
        )

        result = results[0]

        # 生成可视化（如果需要）
        visualization_base64 = None
        if request.return_visualization:
            vis_image = RexOmniVisualize(
                image=img,
                predictions=result["extracted_predictions"],
                font_size=20,
                draw_width=2,  # 更细的线条
                show_labels=True,
                font_path=config.font_path  # 支持中文标签
            )
            visualization_base64 = image_to_base64(vis_image)

        return JSONResponse(content={
            "success": True,
            "task": result["task"],
            "predictions": result["extracted_predictions"],
            "raw_output": result["raw_output"],
            "image_size": result["image_size"],
            "inference_time": time.time() - start_time,
            "visualization": visualization_base64
        })

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的 JSON 参数")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/api/detect_base64")
async def detect_objects_base64(
    image_base64: str = Form(..., description="Base64 编码的图像"),
    request_json: str = Form(..., description="JSON 格式的检测参数")
):
    """
    目标检测端点（Base64 版本）

    接收 base64 编码的图像和检测参数，返回检测结果
    """
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # 解析请求参数
        request_data = json.loads(request_json)
        request = DetectionRequest(**request_data)

        # 加载图像
        img = load_image_from_base64(image_base64)

        # 执行检测
        start_time = time.time()
        results = rex_model.inference(
            images=img,
            task=request.task,
            categories=request.categories,
            keypoint_type=request.keypoint_type,
            visual_prompt_boxes=request.visual_prompt_boxes
        )

        result = results[0]

        # 生成可视化（如果需要）
        visualization_base64 = None
        if request.return_visualization:
            vis_image = RexOmniVisualize(
                image=img,
                predictions=result["extracted_predictions"],
                font_size=20,
                draw_width=2,  # 更细的线条
                show_labels=True,
                font_path=config.font_path  # 支持中文标签
            )
            visualization_base64 = image_to_base64(vis_image)

        return JSONResponse(content={
            "success": True,
            "task": result["task"],
            "predictions": result["extracted_predictions"],
            "raw_output": result["raw_output"],
            "image_size": result["image_size"],
            "inference_time": time.time() - start_time,
            "visualization": visualization_base64
        })

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的 JSON 参数")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.get("/api/tasks")
async def get_supported_tasks():
    """获取支持的任务列表"""
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    tasks = rex_model.get_supported_tasks()
    task_info = {}

    for task in tasks:
        try:
            info = rex_model.get_task_info(task)
            task_info[task] = info
        except:
            pass

    return JSONResponse(content={
        "success": True,
        "tasks": tasks,
        "task_details": task_info
    })


# ==================== 便捷接口（简化的任务特定端点） ====================

@app.post("/api/keypoint")
async def detect_keypoint(
    image: UploadFile = File(..., description="要检测的图像文件"),
    keypoint_type: str = Form("person", description="关键点类型: person, hand, animal"),
    categories: str = Form(None, description="检测类别，多个用逗号分隔（可选，默认使用keypoint_type）"),
    return_visualization: bool = Form(True, description="是否返回可视化图像")
):
    """
    关键点检测接口

    支持的关键点类型:
    - person: 人体关键点（17个关键点）
    - hand: 手部关键点
    - animal: 动物关键点

    返回关键点坐标及可视化图像（带骨架连接）
    """
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        img = load_image_from_upload(image)
        w, h = img.size
        start_time = time.time()

        # 处理 categories
        if categories:
            categories_list = [cat.strip() for cat in categories.split(",") if cat.strip()]
        else:
            categories_list = [keypoint_type]

        # 执行关键点检测
        results = rex_model.inference(
            images=img,
            task="keypoint",
            categories=categories_list,
            keypoint_type=keypoint_type
        )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化（如果需要）
        visualization_base64 = None
        if return_visualization:
            vis_image = RexOmniVisualize(
                image=img,
                predictions=predictions,
                font_size=20,
                draw_width=2,  # 更细的线条
                show_labels=True,
                font_path=config.font_path  # 支持中文标签
            )
            visualization_base64 = image_to_base64(vis_image)

        return JSONResponse({
            "success": True,
            "task": "keypoint",
            "keypoint_type": keypoint_type,
            "predictions": predictions,
            "visualization": visualization_base64,
            "image_size": [w, h],
            "inference_time": time.time() - start_time,
            "raw_output": result["raw_output"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"关键点检测失败: {str(e)}")


@app.post("/api/ocr")
async def detect_ocr(
    image: UploadFile = File(..., description="要识别的图像文件"),
    output_format: str = Form("box", description="输出格式: box (边界框) 或 polygon (多边形)"),
    granularity: str = Form("word", description="粒度: word (单词级别) 或 text_line (文本行级别)"),
    return_visualization: bool = Form(True, description="是否返回可视化图像")
):
    """
    OCR 文字识别接口

    参数:
    - output_format: "box" 或 "polygon"
    - granularity: "word" 或 "text_line"

    返回识别的文字及位置信息
    """
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        img = load_image_from_upload(image)
        w, h = img.size
        start_time = time.time()

        # 确定任务类型
        if output_format.lower() == "polygon":
            task = "ocr_polygon"
        else:
            task = "ocr_box"

        # 确定类别（粒度）
        if granularity.lower() == "text_line" or granularity.lower() == "text line":
            categories = ["text line"]
        else:
            categories = ["word"]

        # 执行 OCR
        results = rex_model.inference(
            images=img,
            task=task,
            categories=categories
        )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化（如果需要）
        visualization_base64 = None
        if return_visualization:
            vis_image = RexOmniVisualize(
                image=img,
                predictions=predictions,
                font_size=20,
                draw_width=2,  # 更细的线条
                show_labels=True,
                font_path=config.font_path  # 支持中文标签
            )
            visualization_base64 = image_to_base64(vis_image)

        return JSONResponse({
            "success": True,
            "task": task,
            "output_format": output_format,
            "granularity": granularity,
            "predictions": predictions,
            "visualization": visualization_base64,
            "image_size": [w, h],
            "inference_time": time.time() - start_time,
            "raw_output": result["raw_output"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(e)}")


@app.post("/api/pointing")
async def detect_pointing(
    image: UploadFile = File(..., description="要检测的图像文件"),
    categories: str = Form(..., description="要指向的目标类别，多个用逗号分隔"),
    return_visualization: bool = Form(True, description="是否返回可视化图像")
):
    """
    Pointing 指向任务接口

    用于定位目标对象的中心点或交互区域
    例如: "where can I hold the cup" -> 返回可以抓握的点

    参数:
    - categories: 目标类别，如 "cup", "door handle" 等
    """
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        img = load_image_from_upload(image)
        w, h = img.size
        start_time = time.time()

        # 处理类别
        categories_list = [cat.strip() for cat in categories.split(",") if cat.strip()]
        if not categories_list:
            raise HTTPException(status_code=400, detail="必须提供至少一个类别")

        # 执行 Pointing 检测
        results = rex_model.inference(
            images=img,
            task="pointing",
            categories=categories_list
        )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化（如果需要）
        visualization_base64 = None
        if return_visualization:
            vis_image = RexOmniVisualize(
                image=img,
                predictions=predictions,
                font_size=20,
                draw_width=2,  # 更细的线条
                show_labels=True,
                font_path=config.font_path  # 支持中文标签
            )
            visualization_base64 = image_to_base64(vis_image)

        return JSONResponse({
            "success": True,
            "task": "pointing",
            "predictions": predictions,
            "visualization": visualization_base64,
            "image_size": [w, h],
            "inference_time": time.time() - start_time,
            "raw_output": result["raw_output"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pointing检测失败: {str(e)}")


@app.post("/api/visual_prompting")
async def detect_visual_prompting(
    image: UploadFile = File(..., description="要检测的图像文件"),
    visual_prompt_boxes: str = Form(..., description="视觉提示框，JSON格式的坐标数组: [[x0,y0,x1,y1], ...]"),
    return_visualization: bool = Form(True, description="是否返回可视化图像")
):
    """
    Visual Prompting 视觉提示接口

    通过提供示例框（visual prompt boxes），模型会在图像中找到相似的对象

    参数:
    - visual_prompt_boxes: JSON格式的边界框列表，如 "[[100,100,200,200], [300,300,400,400]]"
    """
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        img = load_image_from_upload(image)
        w, h = img.size
        start_time = time.time()

        # 解析 visual_prompt_boxes
        try:
            boxes = json.loads(visual_prompt_boxes)
            if not isinstance(boxes, list) or len(boxes) == 0:
                raise ValueError("visual_prompt_boxes 必须是非空数组")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="visual_prompt_boxes 格式错误，必须是JSON数组")

        # 执行 Visual Prompting
        results = rex_model.inference(
            images=img,
            task="visual_prompting",
            categories=["object"],  # Visual prompting 不需要显式类别
            visual_prompt_boxes=boxes
        )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化（如果需要）
        visualization_base64 = None
        if return_visualization:
            vis_image = RexOmniVisualize(
                image=img,
                predictions=predictions,
                font_size=20,
                draw_width=2,  # 更细的线条
                show_labels=True,
                font_path=config.font_path  # 支持中文标签
            )
            visualization_base64 = image_to_base64(vis_image)

        return JSONResponse({
            "success": True,
            "task": "visual_prompting",
            "predictions": predictions,
            "visualization": visualization_base64,
            "image_size": [w, h],
            "inference_time": time.time() - start_time,
            "raw_output": result["raw_output"]
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual Prompting失败: {str(e)}")


@app.post("/api/detect_for_chat")
async def detect_for_chat(
    image: UploadFile = File(..., description="要检测的图像文件"),
    categories: str = Form(..., description="检测类别，多个用逗号分隔"),
    return_visualization: bool = Form(True, description="是否返回可视化图像"),
    show_labels: bool = Form(True, description="是否在可视化图像上显示标签")
):
    """
    简化的检测接口，专用于与其他服务集成

    固定使用 Detection 任务，只需要提供 categories 参数

    参数:
    - categories: 检测类别，如 "person, car, dog"
    - return_visualization: 是否返回可视化图像（默认 True）
    - show_labels: 是否在可视化图像上显示标签（默认 True）

    返回:
    - visualization: base64 编码的可视化图像（带检测框）
    - detection_results: 检测结果
    - image_size: 图像尺寸
    - inference_time: 推理耗时
    """
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        img = load_image_from_upload(image)
        w, h = img.size
        start_time = time.time()



        # 处理类别
        categories_list = [cat.strip() for cat in categories.split(",") if cat.strip()]
        if not categories_list:
            raise HTTPException(status_code=400, detail="必须提供至少一个类别")
        # 使用 Detection 任务进行检测
        results = rex_model.inference(
            images=img,
            task="detection",
            categories=categories_list
        )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化
        visualization_base64 = None
        if return_visualization:

            vis_image = RexOmniVisualize(
                image=img,
                predictions=predictions,
                font_size=20,
                draw_width=2,  # 更细的线条
                show_labels=show_labels,  # 可配置是否显示标签
                font_path=config.font_path  # 支持中文标签
            )
            visualization_base64 = image_to_base64(vis_image)

        return JSONResponse({
            "success": True,
            "visualization": visualization_base64,
            "detection_results": predictions,
            "image_size": [w, h],
            "inference_time": time.time() - start_time,
            "raw_output": result["raw_output"]
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


# ==================== 启动函数 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Rex-Omni API 服务器")
    parser.add_argument(
        "--model_path",
        default="IDEA-Research/Rex-Omni",
        help="模型路径或 HuggingFace 仓库 ID"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="推理后端"
    )
    parser.add_argument("--quantization", type=str, default=None, help="量化类型（如 awq）")
    parser.add_argument("--font_path", type=str, default="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                        help="中文字体路径（用于可视化中文标签）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)

    return parser.parse_args()


def main():
    global rex_model

    args = parse_args()

    # 设置中文字体路径
    config.font_path = args.font_path

    print("🚀 正在初始化 Rex-Omni 模型...")
    print(f"模型: {args.model_path}")
    print(f"后端: {args.backend}")
    print(f"字体路径: {config.font_path}")

    # 初始化模型
    rex_model = RexOmniWrapper(
        model_path=args.model_path,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        quantization=args.quantization,
    )

    print("✅ 模型初始化成功！")
    print(f"🌐 启动 API 服务器: http://{args.host}:{args.port}")
    print(f"📚 API 文档: http://{args.host}:{args.port}/docs")

    # 启动服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
