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
import asyncio
import os
import re
import uuid
import traceback
from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import Any, List, Optional, Union, Tuple

import uvicorn
import httpx  # 用于异步下载图片
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

# 尝试导入 MinIO，如果没有安装则降级运行
try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    Minio = None
    MINIO_AVAILABLE = False
    print("⚠️ 未检测到 minio 库，MinIO 相关功能将不可用。请运行: pip install minio")

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
    image_url: Optional[str] = Field(
        default=None,
        description="图片 URL，支持 http/https (优先级: File > Base64 > URL)"
    )
    upload_result_to_minio: bool = Field(
        default=False,
        description="是否将可视化结果上传到 MinIO 并返回 URL"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    backend: str
    supported_tasks: List[str]
    minio_connected: bool


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
        
        # MinIO 配置
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        self.minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "rex-omni")
        # 专门存放注册物体的样张路径
        self.prototype_prefix = "prototypes"

# 全局配置实例
config = AppConfig()

app = FastAPI(
    title="Rex-Omni API",
    description="目标检测 API 服务 (支持 URL/Base64/File 及 并发控制)",
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

# 全局异步锁：确保 GPU 推理时的线程安全，防止并发导致显存溢出
model_lock = asyncio.Lock()

# 全局 MinIO 客户端
minio_client: Optional[Any] = None

# 全局样张缓存 
# Key: object_name, Value: PIL.Image
PROTOTYPE_CACHE = {}


# ==================== MinIO 初始化与工具函数 ====================

def init_minio():
    """初始化 MinIO 客户端"""
    global minio_client
    if not MINIO_AVAILABLE:
        return

    try:
        # 解析 Endpoint，移除 http/https 前缀
        endpoint_url = config.minio_endpoint
        secure = True
        if endpoint_url.startswith("http://"):
            endpoint_url = endpoint_url.replace("http://", "")
            secure = False
        elif endpoint_url.startswith("https://"):
            endpoint_url = endpoint_url.replace("https://", "")
            secure = True
            
        minio_client = Minio(
            endpoint_url,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=secure
        )
        # 检查 Bucket 是否存在，不存在则创建
        if not minio_client.bucket_exists(config.minio_bucket_name):
            minio_client.make_bucket(config.minio_bucket_name)
            print(f"📦 Created MinIO bucket: {config.minio_bucket_name}")
            
        print(f"✅ MinIO 连接成功: {config.minio_endpoint} -> {config.minio_bucket_name}")
        
    except Exception as e:
        print(f"⚠️ MinIO 初始化失败: {str(e)}")
        minio_client = None


def upload_image_to_minio_sync(image: Image.Image, prefix: str = "vis_result", filename: str = None) -> dict:
    """
    上传 PIL 图像到 MinIO 并返回可访问的 URL 和 路径信息
    """
    if not minio_client:
        return None
        
    try:
        # 将 PIL 图片转为 Bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        file_size = img_byte_arr.getbuffer().nbytes
        
        # 如果未指定文件名，生成随机名: vis_result/YYYYMMDD/uuid.jpg
        if not filename:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{prefix}/{date_str}/{uuid.uuid4()}.jpg"
        else:
            # 确保包含 prefix (避免重复叠加)
            if not filename.startswith(prefix):
                filename = f"{prefix}/{filename}"
        
        # 上传
        minio_client.put_object(
            config.minio_bucket_name,
            filename,
            img_byte_arr,
            file_size,
            content_type="image/jpeg"
        )
        
        # 生成预签名 URL (有效期 7 天)
        url = minio_client.presigned_get_object(
            config.minio_bucket_name,
            filename,
            expires=timedelta(days=7)
        )
        return {
            "url": url,
            "path": filename,
            "bucket": config.minio_bucket_name
        }
    except Exception as e:
        print(f"MinIO 上传失败: {str(e)}")
        return None


def download_from_minio_sync(bucket: str, object_name: str) -> Image.Image:
    """(同步) 从 MinIO 下载图片"""
    if not minio_client:
        raise Exception("MinIO client not initialized")
    
    response = None
    try:
        response = minio_client.get_object(bucket, object_name)
        image_data = response.read()
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    finally:
        if response:
            response.close()


def get_prototype_from_minio(object_name: str) -> Optional[Image.Image]:
    """从 MinIO 获取注册物体的样张图"""
    if not minio_client:
        return None
    
    # 假设注册时文件名是 object_name.jpg
    filename = f"{config.prototype_prefix}/{object_name}.jpg"
    
    try:
        # 检查文件是否存在
        minio_client.stat_object(config.minio_bucket_name, filename)
        # 下载
        return download_from_minio_sync(config.minio_bucket_name, filename)
    except Exception:
        # 文件不存在或其他错误
        return None


# ==================== 图像拼接与解析逻辑 ====================

def stitch_image_for_visual_prompt(target_img: Image.Image, prototype_img: Image.Image) -> Tuple[Image.Image, List[float], int]:
    """
    将样张拼在目标图上方，用于 One-Shot Visual Prompting
    返回: (stitched_img, prompt_box=[x1,y1,x2,y2], offset_y)
    """
    p_w, p_h = prototype_img.size
    t_w, t_h = target_img.size
    
    # 缩放样张：宽度不超过目标图，高度限制在合理范围 (30%)
    scale = 1.0
    if p_w > t_w:
        scale = t_w / p_w
    if p_h * scale > t_h * 0.3:
        scale = min(scale, (t_h * 0.3) / p_h)
        
    if scale != 1.0:
        new_size = (int(p_w * scale), int(p_h * scale))
        new_size = (max(1, new_size[0]), max(1, new_size[1]))
        prototype_img = prototype_img.resize(new_size, Image.Resampling.LANCZOS)
        p_w, p_h = new_size
        
    margin = 10
    new_w = max(t_w, p_w)
    new_h = t_h + p_h + margin
    
    # 使用灰色背景填充
    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
    canvas.paste(prototype_img, (0, 0))
    
    offset_y = p_h + margin
    canvas.paste(target_img, (0, offset_y))
    
    # 样张的 Prompt Box
    prompt_box = [0.0, 0.0, float(p_w), float(p_h)]
    print(f"Stitch Info: OffsetY={offset_y}, TotalH={new_h}")
    return canvas, prompt_box, offset_y


def parse_rex_omni_raw_output(raw_output: str, width: int, height: int, offset_y: int = 0) -> List[dict]:
    """
    解析 Rex-Omni 的原始输出 (支持 Referring Expression 格式)
    格式示例: <|object_ref_start|>object_1...<|box_start|><x1><y1><x2><y2>...
    """
    print(f"🔍 Parsing Raw Output... Output Length: {len(raw_output)}")
    predictions = []
    
    try:
        # 提取所有 <num><num><num><num> 组合
        coords_groups = re.findall(r"<(\d+)><(\d+)><(\d+)><(\d+)>", raw_output)
        
        if not coords_groups:
            print("No coordinates found in raw output")
            return []

        print(f"Found {len(coords_groups)} coordinate groups")

        for i, (c1, c2, c3, c4) in enumerate(coords_groups):
            # Rex-Omni / Shikra 标准坐标顺序: [x1, y1, x2, y2]
            n_x1, n_y1, n_x2, n_y2 = int(c1), int(c2), int(c3), int(c4)
            
            # 还原到拼接图的像素坐标 (0-1000 -> Pixel)
            x1 = (n_x1 / 1000.0) * width
            y1 = (n_y1 / 1000.0) * height
            x2 = (n_x2 / 1000.0) * width
            y2 = (n_y2 / 1000.0) * height
            
            # 计算中心点 Y
            cy = (y1 + y2) / 2
            
            # [关键过滤] 如果框的中心在偏移量之上（即在样张区域），则是 Prompt 本身，跳过
            if cy < (offset_y - 5):
                print(f" Box {i} (Prompt): Ignored (In Prototype Area)")
                continue
                
            # [关键还原] 减去偏移量，变回原图坐标
            orig_y1 = max(0, y1 - offset_y)
            orig_y2 = max(0, y2 - offset_y)
            
            # 检查有效性
            if orig_y2 <= orig_y1:
                print(f" Box {i} (Invalid): Height <= 0 after offset adjustment")
                continue
                
            print(f"  Box {i} (Valid): {x1:.1f}, {orig_y1:.1f}, {x2:.1f}, {orig_y2:.1f}")
            
            predictions.append({
                "box": [x1, orig_y1, x2, orig_y2],
                "score": 1.0,
                "label": "object"
            })
            
    except Exception as e:
        print(f"Parsing Error: {e}")
        traceback.print_exc()
        
    return predictions


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


async def load_image_from_url(url: str) -> Image.Image:
    """从 URL 异步下载图像（增强版：支持 HTTP 和 MinIO 内部鉴权下载）"""
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="无效的图片 URL 协议")
    
    # 1. 尝试普通 HTTP 下载 (verify=False 忽略 SSL 错误，兼容自签名证书)
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            pass # 如果失败，继续尝试 MinIO 备用逻辑

    # 2. 如果 HTTP 失败，检查是否是内部 MinIO 资源
    # 去除 http/https 后比较 host
    endpoint_host = config.minio_endpoint.replace("https://", "").replace("http://", "")
    
    if minio_client and (endpoint_host in url):
        try:
            # 假设 URL 格式: https://host:port/bucket/object_path
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/", 1)
            if len(path_parts) == 2:
                bucket, obj_name = path_parts
                # 使用 run_in_threadpool 运行同步的 MinIO 下载
                return await run_in_threadpool(download_from_minio_sync, bucket, obj_name)
        except Exception as e:
            print(f"MinIO 备用下载失败: {e}")

    raise HTTPException(status_code=400, detail=f"无法从 URL 下载图像")


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
        "supported_tasks": rex_model.get_supported_tasks() if rex_model else [],
        "minio_connected": minio_client is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy" if rex_model is not None else "model_not_loaded",
        "model_loaded": rex_model is not None,
        "backend": rex_model.backend if rex_model else "unknown",
        "supported_tasks": rex_model.get_supported_tasks() if rex_model else [],
        "minio_connected": minio_client is not None
    }


@app.post("/api/register_object")
async def register_object(
    image: Optional[UploadFile] = File(None, description="包含新物体的原图"),
    object_name: str = Form(..., description="新物体名称 (唯一ID)"),
    box: str = Form(..., description="物体坐标 [x1,y1,x2,y2] (JSON string)"),
    image_url: Optional[str] = Form(None, description="原图 URL (可选)")
):
    """
    注册新物体
    
    功能:
    1. 接收一张图片和一个坐标框。
    2. 裁剪出该坐标框内的物体。
    3. 将其作为"标准样张"保存到 MinIO 的 prototypes 目录下。
    4. 之后可以通过 object_name 引用它。
    """
    if not minio_client:
        raise HTTPException(status_code=503, detail="MinIO 未连接，无法使用注册功能")
    
    try:
        # 1. 加载图片
        img: Optional[Image.Image] = None
        if image:
            img = load_image_from_upload(image)
        elif image_url:
            img = await load_image_from_url(image_url)
        else:
            raise HTTPException(status_code=400, detail="必须提供图片")

        # 2. 解析坐标
        try:
            box_coords = json.loads(box)
            if len(box_coords) != 4:
                raise ValueError
        except:
             raise HTTPException(status_code=400, detail="box 必须是 [x1, y1, x2, y2] 格式")

        # 3. 裁剪物体 (Crop)
        # 增加一点 padding 可能会更好，这里先严格裁剪
        crop_img = img.crop((box_coords[0], box_coords[1], box_coords[2], box_coords[3]))
        
        # 4. 保存到 MinIO (prototypes/object_name.jpg)
        filename = f"{object_name}.jpg"
        
        result_info = await run_in_threadpool(
            upload_image_to_minio_sync, 
            crop_img, 
            config.prototype_prefix, 
            filename
        )
        
        if not result_info:
            raise Exception("上传 MinIO 失败")
        
        # 注册成功后加入缓存
        PROTOTYPE_CACHE[object_name] = crop_img
        
        return {
            "success": True,
            "message": f"物体 '{object_name}' 已注册",
            "prototype_url": result_info['url'],
            "prototype_path": result_info['path']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")


@app.get("/api/list_registered_objects")
async def list_registered_objects():
    """列出已注册的自定义物体"""
    if not minio_client:
         raise HTTPException(status_code=503, detail="MinIO 未连接")
    
    try:
        objects = minio_client.list_objects(config.minio_bucket_name, prefix=config.prototype_prefix, recursive=True)
        result = []
        for obj in objects:
            # 提取文件名作为 object_name (去除后缀)
            name = obj.object_name.split("/")[-1].split(".")[0]
            result.append(name)
        return {"success": True, "registered_objects": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/detect_for_chat")
async def detect_for_chat(
    image: Optional[UploadFile] = File(None, description="要检测的图像文件"),
    categories: str = Form(..., description="检测类别，多个用逗号分隔"),
    return_visualization: bool = Form(True, description="是否返回可视化图像"),
    show_labels: bool = Form(True, description="是否在可视化图像上显示标签"),
    image_url: Optional[str] = Form(None, description="图片 URL (可选)"),
    upload_result_to_minio: bool = Form(False, description="是否将结果上传到 MinIO (推荐)"),
    use_registered_objects: bool = Form(False, description="是否尝试使用已注册的物体样张进行视觉提示检测"),
):
    """
    简化的检测接口，专用于与其他服务集成
    
    [核心逻辑]
    1. 接收图片和类别。
    2. 如果 use_registered_objects 为 True，尝试在 MinIO/Cache 中查找同名的注册物体样张。
    3. 如果找到，切换到 Visual Prompting 模式；否则使用普通 Detection 模式。
    4. 对结果进行统一格式处理，确保返回结构一致，并支持 MinIO 上传。
    """
    if rex_model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 1. 图像加载：优先使用 File，其次使用 URL
        img: Optional[Image.Image] = None
        
        if image is not None and image.filename:
            try:
                img = load_image_from_upload(image)
            except: pass
        
        if img is None and image_url:
            img = await load_image_from_url(image_url)
            
        if img is None:
             raise HTTPException(status_code=400, detail="必须提供图像 (请上传有效文件 或 填写 image_url)")

        w, h = img.size
        start_time = time.time()

        categories_list = [cat.strip() for cat in categories.split(",") if cat.strip()]
        if not categories_list:
            raise HTTPException(status_code=400, detail="必须提供至少一个类别")
        
        # [修改] 智能识别注册物体 (拼接方案)
        inference_img = img
        final_task = "detection"
        used_prototype_name = None
        stitch_offset_y = 0
        vp_boxes = None
        
        if use_registered_objects and categories_list:
            # 只取第一个类别尝试查找样张
            primary_cat = categories_list[0]
            
            # 检查缓存
            prototype_img = None
            if primary_cat in PROTOTYPE_CACHE:
                prototype_img = PROTOTYPE_CACHE[primary_cat]
            else:
                prototype_img = await run_in_threadpool(get_prototype_from_minio, primary_cat)
                if prototype_img:
                    PROTOTYPE_CACHE[primary_cat] = prototype_img
            
            if prototype_img:
                print(f" Mode Switch: Visual Prompting with '{primary_cat}'")
                
                # 执行拼接
                stitched_img, prompt_box, offset_y = stitch_image_for_visual_prompt(img, prototype_img)
                
                # 更新推理参数
                inference_img = stitched_img
                final_task = "visual_prompting"
                vp_boxes = [prompt_box] # Prompt Box
                categories_list = ["object"] # Visual prompting 不需要具体类别名
                
                used_prototype_name = primary_cat
                stitch_offset_y = offset_y
        
        # 2. 执行检测 (使用锁 + 线程池)
        async with model_lock:
            kwargs = {
                "images": inference_img,
                "task": final_task,
                "categories": categories_list
            }
            
            if final_task == "visual_prompting":
                kwargs["visual_prompt_boxes"] = vp_boxes
            
            print(f"Running Inference: task={final_task}, boxes={vp_boxes}")
            results = await run_in_threadpool(rex_model.inference, **kwargs)
            
        result = results[0]
        raw_output = result.get("raw_output", "")
        raw_preds = result.get("extracted_predictions", [])
        
        # 3. 后处理与格式统一
        # 统一将结果格式化为 Detection 格式 (Dict: {'类别': [List]})
        final_preds_dict = {}
        
        if final_task == "visual_prompting":
             # 强制手动解析 (Force Manual Parsing)
             if raw_output:
                 print("🔧 Visual Prompting: Forcing manual parsing...")
                 parsed_list = parse_rex_omni_raw_output(
                    raw_output, 
                    inference_img.width, 
                    inference_img.height,
                    stitch_offset_y
                )
             else:
                 parsed_list = []
                 
             # 转换为 Dict 格式 {'注册物体名': [box1, box2]}
             if used_prototype_name:
                 box_list = []
                 for p in parsed_list:
                     box_list.append({
                         "type": "box",
                         "coords": p["box"],
                         "score": p.get("score", 1.0)
                     })
                 final_preds_dict = {used_prototype_name: box_list}
        else:
            # 普通模式
            if isinstance(raw_preds, list):
                final_preds_dict = {"object": raw_preds}
            else:
                final_preds_dict = raw_preds

        # 4. 生成可视化 & MinIO 上传
        visualization_base64 = None
        visualization_url = None
        visualization_path = None
        
        # 只要请求了返回图 OR 请求了上传，就执行画图
        if return_visualization or (upload_result_to_minio and minio_client):
            def run_viz():
                return RexOmniVisualize(
                    image=img, # 必须用原图画框
                    predictions=final_preds_dict,
                    font_size=20,
                    draw_width=2,
                    show_labels=show_labels,
                    font_path=config.font_path
                )
            
            try:
                vis_img = await run_in_threadpool(run_viz)
                
                # 逻辑独立：如果需要上传，则上传
                if upload_result_to_minio and minio_client:
                    try:
                        up_res = await run_in_threadpool(upload_image_to_minio_sync, vis_img)
                        if up_res: 
                            visualization_url = up_res['url']
                            visualization_path = up_res['path']
                    except Exception as e:
                        print(f"MinIO Upload Error: {e}")
                
                # 逻辑独立：如果需要返回 Base64，则转换
                if return_visualization:
                    visualization_base64 = image_to_base64(vis_img)
                    
            except Exception as e:
                print(f"Visualization Error: {e}")
                traceback.print_exc()

        return JSONResponse({
            "success": True,
            "mode": final_task,
            "used_prototype": used_prototype_name,
            "visualization": visualization_base64,
            "visualization_url": visualization_url,
            "visualization_path": visualization_path,
            "detection_results": final_preds_dict,
            "image_size": [w, h],
            "inference_time": time.time() - start_time,
            "raw_output": raw_output
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


# ==================== 其他端点 (检测/OCR/Keypoint) ====================

@app.post("/api/detect")
async def detect_objects(
    image: Optional[UploadFile] = File(None, description="要检测的图像文件"),
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
        request_data = json.loads(request_json)
        request = DetectionRequest(**request_data)

        img = Optional[Image.Image] = None
        if image is not None and image.filename:
            img = load_image_from_upload(image)
        elif request.image_url:
            img = await load_image_from_url(request.image_url)
        else:
             raise HTTPException(status_code=400, detail="必须提供图像文件或 image_url")

        start_time = time.time()
        
        async with model_lock:
            results = await run_in_threadpool(
                rex_model.inference,
                images=img,
                task=request.task,
                categories=request.categories,
                keypoint_type=request.keypoint_type,
                visual_prompt_boxes=request.visual_prompt_boxes
            )

        result = results[0]

        # 生成可视化 & MinIO 上传 (逻辑解耦)
        visualization_base64 = None
        visualization_url = None
        visualization_path = None
        
        if request.return_visualization or (request.upload_result_to_minio and minio_client):
            def run_viz():
                return RexOmniVisualize(
                    image=img,
                    predictions=result["extracted_predictions"],
                    font_size=20,
                    draw_width=2,
                    show_labels=True,
                    font_path=config.font_path
                )
            
            try:
                vis_image = await run_in_threadpool(run_viz)
                
                if request.upload_result_to_minio and minio_client:
                    try:
                        up_res = await run_in_threadpool(upload_image_to_minio_sync, vis_image)
                        if up_res:
                            visualization_url = up_res['url']
                            visualization_path = up_res['path']
                    except Exception as e:
                        print(f"MinIO Upload Error in detect: {e}")

                if request.return_visualization:
                    visualization_base64 = image_to_base64(vis_image)
            except Exception as e:
                print(f"Visualization Error in detect: {e}")

        return JSONResponse(content={
            "success": True,
            "task": result["task"],
            "predictions": result["extracted_predictions"],
            "raw_output": result["raw_output"],
            "image_size": result["image_size"],
            "inference_time": time.time() - start_time,
            "visualization": visualization_base64,
            "visualization_url": visualization_url,
            "visualization_path": visualization_path
        })

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的 JSON 参数")
    except HTTPException:
        raise
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
        request_data = json.loads(request_json)
        request = DetectionRequest(**request_data)

        img = load_image_from_base64(image_base64)

        start_time = time.time()
        
        async with model_lock:
            results = await run_in_threadpool(
                rex_model.inference,
                images=img,
                task=request.task,
                categories=request.categories,
                keypoint_type=request.keypoint_type,
                visual_prompt_boxes=request.visual_prompt_boxes
            )

        result = results[0]

        # 生成可视化 & MinIO 上传 (逻辑解耦)
        visualization_base64 = None
        visualization_url = None
        visualization_path = None
        
        if request.return_visualization or (request.upload_result_to_minio and minio_client):
            def run_viz():
                return RexOmniVisualize(
                    image=img,
                    predictions=result["extracted_predictions"],
                    font_size=20,
                    draw_width=2,
                    show_labels=True,
                    font_path=config.font_path
                )
            
            try:
                vis_image = await run_in_threadpool(run_viz)
                
                if request.upload_result_to_minio and minio_client:
                    try:
                        up_res = await run_in_threadpool(upload_image_to_minio_sync, vis_image)
                        if up_res:
                            visualization_url = up_res['url']
                            visualization_path = up_res['path']
                    except Exception as e:
                        print(f"MinIO Upload Error in detect_base64: {e}")

                if request.return_visualization:
                    visualization_base64 = image_to_base64(vis_image)
            except Exception as e:
                print(f"Visualization Error in detect_base64: {e}")

        return JSONResponse(content={
            "success": True,
            "task": result["task"],
            "predictions": result["extracted_predictions"],
            "raw_output": result["raw_output"],
            "image_size": result["image_size"],
            "inference_time": time.time() - start_time,
            "visualization": visualization_base64,
            "visualization_url": visualization_url,
            "visualization_path": visualization_path
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
    return_visualization: bool = Form(True, description="是否返回可视化图像"),
    upload_result_to_minio: bool = Form(False, description="是否将结果上传到 MinIO")
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

        if categories:
            categories_list = [cat.strip() for cat in categories.split(",") if cat.strip()]
        else:
            categories_list = [keypoint_type]

        async with model_lock:
            results = await run_in_threadpool(
                rex_model.inference,
                images=img,
                task="keypoint",
                categories=categories_list,
                keypoint_type=keypoint_type
            )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化 & MinIO 上传 (逻辑解耦)
        visualization_base64 = None
        visualization_url = None
        visualization_path = None
        
        if return_visualization or (upload_result_to_minio and minio_client):
            def run_viz():
                return RexOmniVisualize(
                    image=img,
                    predictions=predictions,
                    font_size=20,
                    draw_width=2,
                    show_labels=True,
                    font_path=config.font_path
                )
            
            try:
                vis_image = await run_in_threadpool(run_viz)
                
                if upload_result_to_minio and minio_client:
                    try:
                        up_res = await run_in_threadpool(upload_image_to_minio_sync, vis_image)
                        if up_res:
                            visualization_url = up_res['url']
                            visualization_path = up_res['path']
                    except Exception as e:
                        print(f"MinIO Upload Error in keypoint: {e}")

                if return_visualization:
                    visualization_base64 = image_to_base64(vis_image)
            except Exception as e:
                print(f"Visualization Error in keypoint: {e}")

        return JSONResponse({
            "success": True,
            "task": "keypoint",
            "keypoint_type": keypoint_type,
            "predictions": predictions,
            "visualization": visualization_base64,
            "visualization_url": visualization_url,
            "visualization_path": visualization_path,
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
    return_visualization: bool = Form(True, description="是否返回可视化图像"),
    upload_result_to_minio: bool = Form(False, description="是否将结果上传到 MinIO")
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

        if output_format.lower() == "polygon":
            task = "ocr_polygon"
        else:
            task = "ocr_box"

        if granularity.lower() == "text_line" or granularity.lower() == "text line":
            categories = ["text line"]
        else:
            categories = ["word"]

        async with model_lock:
            results = await run_in_threadpool(
                rex_model.inference,
                images=img,
                task=task,
                categories=categories
            )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化 & MinIO 上传 (逻辑解耦)
        visualization_base64 = None
        visualization_url = None
        visualization_path = None
        
        if return_visualization or (upload_result_to_minio and minio_client):
            def run_viz():
                return RexOmniVisualize(
                    image=img,
                    predictions=predictions,
                    font_size=20,
                    draw_width=2,
                    show_labels=True,
                    font_path=config.font_path
                )
            
            try:
                vis_image = await run_in_threadpool(run_viz)
                
                if upload_result_to_minio and minio_client:
                    try:
                        up_res = await run_in_threadpool(upload_image_to_minio_sync, vis_image)
                        if up_res:
                            visualization_url = up_res['url']
                            visualization_path = up_res['path']
                    except Exception as e:
                        print(f" MinIO Upload Error in ocr: {e}")

                if return_visualization:
                    visualization_base64 = image_to_base64(vis_image)
            except Exception as e:
                print(f"Visualization Error in ocr: {e}")

        return JSONResponse({
            "success": True,
            "task": task,
            "output_format": output_format,
            "granularity": granularity,
            "predictions": predictions,
            "visualization": visualization_base64,
            "visualization_url": visualization_url,
            "visualization_path": visualization_path,
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
    return_visualization: bool = Form(True, description="是否返回可视化图像"),
    upload_result_to_minio: bool = Form(False, description="是否将结果上传到 MinIO")
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

        categories_list = [cat.strip() for cat in categories.split(",") if cat.strip()]
        if not categories_list:
            raise HTTPException(status_code=400, detail="必须提供至少一个类别")

        async with model_lock:
            results = await run_in_threadpool(
                rex_model.inference,
                images=img,
                task="pointing",
                categories=categories_list
            )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化 & MinIO 上传 (逻辑解耦)
        visualization_base64 = None
        visualization_url = None
        visualization_path = None
        
        if return_visualization or (upload_result_to_minio and minio_client):
            def run_viz():
                return RexOmniVisualize(
                    image=img,
                    predictions=predictions,
                    font_size=20,
                    draw_width=2,
                    show_labels=True,
                    font_path=config.font_path
                )
            
            try:
                vis_image = await run_in_threadpool(run_viz)
                
                if upload_result_to_minio and minio_client:
                    try:
                        up_res = await run_in_threadpool(upload_image_to_minio_sync, vis_image)
                        if up_res:
                            visualization_url = up_res['url']
                            visualization_path = up_res['path']
                    except Exception as e:
                        print(f"MinIO Upload Error in pointing: {e}")

                if return_visualization:
                    visualization_base64 = image_to_base64(vis_image)
            except Exception as e:
                print(f"Visualization Error in pointing: {e}")

        return JSONResponse({
            "success": True,
            "task": "pointing",
            "predictions": predictions,
            "visualization": visualization_base64,
            "visualization_url": visualization_url,
            "visualization_path": visualization_path,
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
    return_visualization: bool = Form(True, description="是否返回可视化图像"),
    upload_result_to_minio: bool = Form(False, description="是否将结果上传到 MinIO")
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

        try:
            boxes = json.loads(visual_prompt_boxes)
            if not isinstance(boxes, list) or len(boxes) == 0:
                raise ValueError("visual_prompt_boxes 必须是非空数组")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="visual_prompt_boxes 格式错误，必须是JSON数组")

        async with model_lock:
            results = await run_in_threadpool(
                rex_model.inference,
                images=img,
                task="visual_prompting",
                categories=["object"],  # Visual prompting 不需要显式类别
                visual_prompt_boxes=boxes
            )
        result = results[0]
        predictions = result["extracted_predictions"]

        # 生成可视化 & MinIO 上传 (逻辑解耦)
        visualization_base64 = None
        visualization_url = None
        visualization_path = None
        
        if return_visualization or (upload_result_to_minio and minio_client):
            def run_viz():
                return RexOmniVisualize(
                    image=img,
                    predictions=predictions,
                    font_size=20,
                    draw_width=2,
                    show_labels=True,
                    font_path=config.font_path
                )
            
            try:
                vis_image = await run_in_threadpool(run_viz)
                
                if upload_result_to_minio and minio_client:
                    try:
                        up_res = await run_in_threadpool(upload_image_to_minio_sync, vis_image)
                        if up_res:
                            visualization_url = up_res['url']
                            visualization_path = up_res['path']
                    except Exception as e:
                        print(f"MinIO Upload Error in visual_prompting: {e}")

                if return_visualization:
                    visualization_base64 = image_to_base64(vis_image)
            except Exception as e:
                print(f"Visualization Error in visual_prompting: {e}")

        return JSONResponse({
            "success": True,
            "task": "visual_prompting",
            "predictions": predictions,
            "visualization": visualization_base64,
            "visualization_url": visualization_url,
            "visualization_path": visualization_path,
            "image_size": [w, h],
            "inference_time": time.time() - start_time,
            "raw_output": result["raw_output"]
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual Prompting失败: {str(e)}")


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
    parser.add_argument("--port", type=int, default=8001, help="服务器端口")
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
    
    # 初始化 MinIO
    init_minio()

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