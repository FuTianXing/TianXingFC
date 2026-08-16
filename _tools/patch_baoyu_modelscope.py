# -*- coding: utf-8 -*-
"""给 baoyu-image-gen v2.1.0 的 openai provider 加 ModelScope(魔搭) 异步任务轮询适配。

改动点（与 v1 补丁相同的原理，适配新版函数签名 generateWithOpenAIGenerations(baseURL, apiKey, body)）：
1. baseURL 含 "modelscope" 时，提交请求加 X-ModelScope-Async-Mode: true 头
2. 响应含 task_id 时，轮询 /tasks/{id} 直到 SUCCEED 并下载 output_images[0]
"""
import io
import sys

PATH = r"C:\Users\Administrator\.agents\skills\baoyu-image-gen\scripts\providers\openai.ts"

OLD = '''  const res = await fetch(`${baseURL}/images/generations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI API error: ${err}`);
  }

  const result = (await res.json()) as OpenAIImageResponse;
  return extractImageFromResponse(result);
}'''

NEW = '''  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  };
  // ModelScope(魔搭) 兼容：提交时启用异步任务模式
  if (baseURL.includes("modelscope")) {
    headers["X-ModelScope-Async-Mode"] = "true";
  }

  const res = await fetch(`${baseURL}/images/generations`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI API error: ${err}`);
  }

  const result = (await res.json()) as OpenAIImageResponse & {
    task_id?: string;
    task_status?: string;
  };

  // ModelScope 异步任务协议：提交只返回 task_id，需要轮询任务状态
  if (result.task_id) {
    return await pollModelScopeTask(baseURL, apiKey, result.task_id);
  }

  return extractImageFromResponse(result);
}

/**
 * ModelScope(魔搭) 异步任务轮询：GET /tasks/{id} 直到 SUCCEED，下载 output_images[0]。
 */
async function pollModelScopeTask(
  baseURL: string,
  apiKey: string,
  taskId: string
): Promise<Uint8Array> {
  const url = `${baseURL}/tasks/${taskId}`;
  for (let attempt = 0; attempt < 60; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 4000));
    const res = await fetch(url, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "X-ModelScope-Task-Type": "image_generation",
      },
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`ModelScope poll error (${res.status}): ${err}`);
    }
    const data = (await res.json()) as {
      task_status?: string;
      output_images?: string[];
      message?: string;
    };
    if (data.task_status === "SUCCEED" && data.output_images?.[0]) {
      const imgUrl = data.output_images[0];
      const imgRes = await fetch(imgUrl);
      if (!imgRes.ok) throw new Error("Failed to download image");
      const buf = await imgRes.arrayBuffer();
      return new Uint8Array(buf);
    }
    if (data.task_status === "FAILED") {
      throw new Error(`ModelScope generation failed: ${data.message || "unknown error"}`);
    }
  }
  throw new Error("ModelScope polling timeout after 4 minutes.");
}'''

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if OLD not in src:
    print("PATTERN NOT FOUND, aborting", file=sys.stderr)
    sys.exit(1)

new_src = src.replace(OLD, NEW, 1)

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(new_src)

print("PATCHED OK")
print("has pollModelScopeTask:", "pollModelScopeTask" in new_src)
print("has X-ModelScope-Async-Mode:", "X-ModelScope-Async-Mode" in new_src)
