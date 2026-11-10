"""Claude 调用封装：图片块构造、结构化输出、工具循环。"""
import base64
import io
import json
import logging
import os
import threading

import anthropic
import numpy as np
from PIL import Image

import config

log = logging.getLogger(__name__)

_client = None
_lock = threading.Lock()


def get_client() -> anthropic.Anthropic:
    global _client
    with _lock:
        if _client is None:
            api_key = os.environ.get(config.API_KEY_ENV)
            if not api_key:
                raise RuntimeError(f"环境变量 {config.API_KEY_ENV} 未设置")
            _client = anthropic.Anthropic(api_key=api_key, base_url=config.BASE_URL, max_retries=4)
        return _client


# ---------------- 内容块 ----------------
def image_block(img) -> dict:
    """numpy(BGR/灰度) 或 PIL 图片 -> base64 image 内容块；过大时转 JPEG。"""
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            pil = Image.fromarray(img)
        else:
            pil = Image.fromarray(img[:, :, ::-1])
    else:
        pil = img
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    media = "image/png"
    if buf.tell() > 3_500_000:
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="JPEG", quality=92)
        media = "image/jpeg"
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


# ---------------- 调用 ----------------
class LLMRefusal(RuntimeError):
    pass


def _create(model, system, messages, effort, schema=None, tools=None):
    client = get_client()
    output_config = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs = dict(
        model=model,
        max_tokens=config.MAX_TOKENS,
        system=system,
        messages=messages,
        output_config=output_config,
    )
    if tools:
        kwargs["tools"] = tools
    if config.ENABLE_FALLBACKS:
        kwargs["betas"] = [config.FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
        with client.beta.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()
    else:
        with client.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()
    if resp.stop_reason == "refusal":
        raise LLMRefusal(f"模型拒答: {getattr(resp, 'stop_details', None)}")
    return resp


def _final_text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


def call_json(model: str, system: str, content: list, schema: dict, effort: str) -> dict:
    """单轮调用并返回结构化 JSON。"""
    messages = [{"role": "user", "content": content}]
    resp = _create(model, system, messages, effort, schema=schema)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("输出被 max_tokens 截断")
    return json.loads(_final_text(resp))


def call_with_tools(model: str, system: str, content: list, schema: dict, effort: str,
                    tools: list, handler, max_turns: int = 12) -> dict:
    """手动工具循环；handler(name, input) -> str。最终返回结构化 JSON。"""
    messages = [{"role": "user", "content": content}]
    for _ in range(max_turns):
        resp = _create(model, system, messages, effort, schema=schema, tools=tools)
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                try:
                    out = handler(block.name, block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
                except Exception as e:  # 工具出错返回给模型自行修正
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": f"ERROR: {e}", "is_error": True})
            messages.append({"role": "user", "content": results})
            continue
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("输出被 max_tokens 截断")
        return json.loads(_final_text(resp))
    raise RuntimeError("工具调用轮数超限")
