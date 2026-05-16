from flask import Flask, request, jsonify
import os
import re
import json

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

@app.route('/')
def index():
    return '血壓記錄 API 運作中 ✅'

@app.route('/api/read_bp', methods=['POST', 'OPTIONS'])
def read_bp():
    if request.method == 'OPTIONS':
        return '', 204

    import requests as req
    import time

    GEMINI_KEY = os.environ.get('GEMINI_API_KEY', '')
    if not GEMINI_KEY:
        return jsonify({'success': False, 'error': 'API Key 未設定'})

    body = request.json or {}
    image_b64  = body.get('image', '')
    image_type = body.get('type', 'image/jpeg')

    if not image_b64:
        return jsonify({'success': False, 'error': '未收到圖片'})

    def call_gemini(prompt, retry=0):
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": image_type, "data": image_b64}}
                ]
            }],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 150,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        r = req.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}",
            json=payload, timeout=30
        )
        if r.status_code == 429 and retry < 2:
            time.sleep(5)
            return call_gemini(prompt, retry + 1)
        return r

    # 第一階段：識別所有數字
    prompt1 = """這是一張血壓計的照片。
請仔細看清楚螢幕上的數字，找出以下三個數值：
- 收縮壓（高壓/最高/SYS）：通常是最大的數字，90-200 之間
- 舒張壓（低壓/最低/DIA）：中間的數字，60-130 之間
- 脈搏（心跳/脈拍/PR/PUL）：最小的數字，50-120 之間

請只回傳這個格式（不要加其他文字）：
SYS=數字
DIA=數字
PUL=數字

看不清楚的寫 UNCLEAR，看不到螢幕寫 NOSCREEN。"""

    try:
        r1 = call_gemini(prompt1)
        if r1.status_code != 200:
            return jsonify({'success': False, 'error': f'Gemini 錯誤 {r1.status_code}: {r1.text[:100]}'})

        text1 = r1.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        print(f"[read_bp] 第一階段: {text1}")

        if 'NOSCREEN' in text1:
            return jsonify({'success': False, 'error': '看不到血壓計螢幕，請重新拍攝'})

        sys_m = re.search(r'SYS[=:\s]+(\d+)', text1, re.IGNORECASE)
        dia_m = re.search(r'DIA[=:\s]+(\d+)', text1, re.IGNORECASE)
        pul_m = re.search(r'PUL[=:\s]+(\d+)', text1, re.IGNORECASE)
        has_unclear = 'UNCLEAR' in text1.upper()

        # 第二階段：若有不確定的數字，再確認一次
        if has_unclear or not (sys_m and dia_m and pul_m):
            prompt2 = f"""這是同一張血壓計照片。
第一次識別結果：{text1}

請再仔細看一次螢幕，特別注意每個數字區域。
以同樣格式回傳：
SYS=數字
DIA=數字
PUL=數字"""
            r2 = call_gemini(prompt2)
            if r2.status_code == 200:
                text2 = r2.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f"[read_bp] 第二階段: {text2}")
                sys_m = re.search(r'SYS[=:\s]+(\d+)', text2, re.IGNORECASE) or sys_m
                dia_m = re.search(r'DIA[=:\s]+(\d+)', text2, re.IGNORECASE) or dia_m
                pul_m = re.search(r'PUL[=:\s]+(\d+)', text2, re.IGNORECASE) or pul_m

        if not (sys_m and dia_m and pul_m):
            return jsonify({'success': False, 'error': '無法識別完整數值，請確保血壓計螢幕清晰可見'})

        sys_val = int(sys_m.group(1))
        dia_val = int(dia_m.group(1))
        pul_val = int(pul_m.group(1))

        # 合理範圍驗證
        errors = []
        if not (60 <= sys_val <= 250): errors.append(f'高壓 {sys_val} 超出合理範圍')
        if not (40 <= dia_val <= 150): errors.append(f'低壓 {dia_val} 超出合理範圍')
        if not (30 <= pul_val <= 200): errors.append(f'脈搏 {pul_val} 超出合理範圍')
        if sys_val <= dia_val:         errors.append(f'高壓應大於低壓')

        if errors:
            return jsonify({'success': False, 'error': '識別異常：' + '、'.join(errors) + '，請重新拍攝'})

        return jsonify({
            'success': True,
            'sys': sys_val,
            'dia': dia_val,
            'pul': pul_val,
            'confidence': 'medium' if has_unclear else 'high',
        })

    except Exception as e:
        print(f"[read_bp] 錯誤: {e}")
        return jsonify({'success': False, 'error': str(e)[:100]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
