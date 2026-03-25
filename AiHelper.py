import asyncio
import aiohttp
from aiohttp import web
from aiohttp_cors import setup, ResourceOptions
import os
import json
import logging
import time
import urllib.parse

async def on_prepare(request, response):
    request.start_time = time.time()

async def on_response(request, response):
    pass

async def on_request_start(request, *args, **kwargs):
    pass

async def on_request_end(request, *args, **kwargs):
    pass

def get_query_param(request, param_name):
    if isinstance(request, aiohttp.web.Request):
        return request.rel_url.query.get(param_name)
    else:
        query_params = urllib.parse.parse_qs(request)
        param_values = query_params.get(param_name, [])
        return param_values[0] if param_values else None

async def get_mjstyle_json(request):
    name = request.match_info['name']
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'docs', 'mjstyle', f'{name}.json')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            json_data = f.read()
        return web.Response(body=json_data, content_type='application/json')
    else:
        return web.Response(status=404)

async def get_marked_js(request):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'web', 'lib', 'marked.min.js')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            js_data = f.read()
        return web.Response(body=js_data, content_type='application/javascript')
    else:
        return web.Response(status=404)

async def get_purify_js(request):
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'web', 'lib', 'purify.min.js')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            js_data = f.read()
        return web.Response(body=js_data, content_type='application/javascript')
    else:
        return web.Response(status=404)

def load_api_config():
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(current_dir, 'Comflyapi.json')
        if not os.path.exists(config_path):
            return {}
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading API config: {str(e)}")
        return {}

def load_billing_config():
    """加载计费配置，支持从挂载的外部配置文件读取"""
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # 优先从环境变量指定的路径读取
        env_path = os.environ.get("BILLING_CONFIG_PATH")
        if env_path and os.path.exists(env_path):
            config_path = env_path
        else:
            # 默认路径
            config_path = os.path.join(current_dir, 'config', 'billing_config.json')

        if not os.path.exists(config_path):
            logging.warning(f"[Billing] Config file not found: {config_path}")
            return {"models": {}, "display_settings": {}}

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"[Billing] Loaded config from: {config_path}")
        return config
    except Exception as e:
        logging.error(f"[Billing] Error loading billing config: {str(e)}")
        return {"models": {}, "display_settings": {}}

async def get_config(request):
    config = load_api_config()
    return web.json_response(config)

async def get_billing_config(request):
    """获取计费配置的 API 端点"""
    config = load_billing_config()
    return web.json_response(config)

def init_server(app):
    app.router.add_get("/api/get_config", get_config)
    app.router.add_get("/api/billing_config", get_billing_config)
    app.router.add_get("/lib/marked.min.js", get_marked_js)
    app.router.add_get("/lib/purify.min.js", get_purify_js)
    app.router.add_get("/mjstyle/{name}.json", get_mjstyle_json)

