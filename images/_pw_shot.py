# -*- coding: utf-8 -*-
"""用 Playwright 渲染本地 HTML 并整页截图，验证信息图布局。"""
import os, sys, asyncio

HTML = r"D:\software\TianXingFC\images\spiral_preview.html"
OUT  = r"D:\software\TianXingFC\images\pw_shot.png"

async def main():
    from playwright.sync_api import sync_playwright  # 不能用，改用 async
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        # 优先用已下载的 chromium，否则用系统 chrome
        browser = None
        try:
            browser = await p.chromium.launch()
            print("used: bundled chromium")
        except Exception as e:
            print("bundled chromium launch failed:", e)
            browser = await p.chromium.launch(
                executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                args=["--disable-gpu", "--no-sandbox"])
            print("used: system chrome")
        page = await browser.new_page(viewport={"width": 1680, "height": 900},
                                      device_scale_factor=1)
        await page.goto("file:///" + HTML.replace("\\", "/"))
        await page.wait_for_timeout(800)
        # 整页截图
        await page.screenshot(path=OUT, full_page=True)
        dim = await page.evaluate("() => ({w: document.body.scrollWidth, h: document.body.scrollHeight})")
        print("page dims:", dim)
        await browser.close()
        print("saved:", OUT, os.path.getsize(OUT))

asyncio.run(main())
