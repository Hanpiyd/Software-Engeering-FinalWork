from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import time
import requests
import sys
import os
from flask import send_from_directory

app = Flask(__name__)
CORS(app)

def ai_communicate(message, max_retries=3, timeout=300):
    url = "https://api.siliconflow.cn/v1/chat/completions"

    headers = {
        "Authorization": "Bearer sk-nlcqvwkkrzdmsunawrxjfrgetbcecntzjqdzcnixuddzmejf",
        "Content-Type": "application/json"
    }

    payload = {
        "model" : "Qwen/Qwen3-8B",
        "messages" : [{
            "role" : "user",
            "content" : f"你是一个智慧海洋牧场的AI助手，拥有丰富的相应各领域的知识。接下来用户将会询问你相关问题，请你耐心、全面、认真地回答:\n{message}"
        }],
        "stream": True,
        "max_tokens": 4096,
        "enable_thinking" : False,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "frequency_penalty": 0.0,
        "n": 1,
        "response_format": {"type": "text"},
    }
    
    for attempt in range(max_retries):
        try:
            print(f"尝试请求 {attempt + 1}/{max_retries}...")
            response = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            return response
            
        except requests.exceptions.Timeout:
            print(f"请求超时，尝试 {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("超过最大重试次数，请求失败")
                raise
        except requests.exceptions.RequestException as e:
            print(f"网络请求异常: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
        except Exception as e:
            print(f"未处理异常: {str(e)}")
            raise

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "请提供消息内容"}), 400
        
        response = ai_communicate(message)
        
        def generate():
            try:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]
                        if line_text == "" or line_text == "[DONE]":
                            continue
                        try:
                            data = json.loads(line_text)
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                                elif "message" in choice and "content" in choice["message"]:
                                    content = choice["message"]["content"]
                                    if content:
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError:
                            print(f"无法解析响应行: {line_text}")
                
                yield f"data: {json.dumps({'content': '', 'finished': True})}\n\n"
            except Exception as e:
                print(f"流处理中发生错误: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists("static/" + path):
        return send_from_directory('static', path)
    else:
        return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port, debug=True)

# R8前置：分支1修改
