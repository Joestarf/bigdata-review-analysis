from __future__ import annotations
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
from .common import ROOT, LABELS, read_json, write_json, write_csv, sha_file, normalize_text
from .annotations import validate_task_manifest, labels_path, GateError, status, require_frozen_guidelines

PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>评论对人工标注</title><style>
body{font-family:system-ui,sans-serif;background:#f2f5f7;color:#182a37;max-width:1100px;margin:30px auto;padding:0 20px}
header,section{background:white;padding:22px;border-radius:12px;margin:15px 0}
h1{font-size:25px}button,select,textarea{font:inherit;padding:10px;border:1px solid #bbc8d0;border-radius:7px}
button{cursor:pointer;background:#e8f3ff}button:disabled{opacity:.45;cursor:default}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.text{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65}
textarea{width:95%;min-height:65px}label{display:inline-block;margin:8px 16px 8px 0}
.small{font-size:14px;color:#536774}.message{color:#9f3622}#pairid{overflow-wrap:anywhere;font-size:12px}
@media(max-width:700px){.columns{grid-template-columns:1fr}}
</style><header><h1>评论对人工标注</h1>
<p>判断两段文字的信息是否重复。相同观点可能来自真实用户，标签不评价评论者身份。</p>
<p class="small">完全重复：正文规范化后相同且内容充分。近似重复：主要内容和细节重复。非重复：包含不同信息或结论。不确定：通用短评或证据不足。</p>
<select id="stage" onchange="loadTask(0)"><option value="pilot">第一批 60 对</option><option value="train">训练标注</option><option value="validation">验证标注</option><option value="test">测试标注（封存用于最终评价）</option><option value="synthetic">合成扰动检查</option></select>
<button onclick="firstBlank()">定位未标注</button><button id="freeze" onclick="freezeGuide()">固定标注规范，解锁后续任务</button>
<p id="progress"></p><p id="message" class="message"></p></header>
<section><p id="pairid"></p><div class="columns"><div><h3>评论 A</h3><div id="a" class="text"></div></div><div><h3>评论 B</h3><div id="b" class="text"></div></div></div></section>
<section><div id="choices">
<label><input type="radio" name="label" value="exact">完全重复</label><label><input type="radio" name="label" value="near">近似重复</label><label><input type="radio" name="label" value="distinct">非重复</label><label><input type="radio" name="label" value="uncertain">不确定</label>
</div><div id="synchoices" hidden><label><input type="radio" name="label" value="approve">变换保留原意</label><label><input type="radio" name="label" value="reject">变换改变原意，需修正</label></div>
<p>写一句判断理由：</p><textarea id="reason" placeholder="例如：两条正文一致；或：都评价续航，但描述的时长和结论不同。"></textarea>
<p><button onclick="move(-1)">上一对</button> <button onclick="save()">保存并下一对</button> <button onclick="move(1)">下一对</button></p>
<p class="small">页面不展示星级、用户身份、商品或模型分数。测试判断单独保存，训练和阈值调整不读取测试标签。</p></section>
<script>
const token='__TOKEN__';let current=0,total=0,task=null;
function message(t){document.getElementById('message').textContent=t}
async function request(url,body){let r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)}:{});let v=await r.json();if(!r.ok)throw Error(v.error);return v}
async function loadTask(i){try{let stage=document.getElementById('stage').value;let v=await request('/api/task?stage='+stage+'&index='+i);task=v;current=v.index;total=v.total;
document.getElementById('a').textContent=v.text_a;document.getElementById('b').textContent=v.text_b;document.getElementById('pairid').textContent='编号：'+v.pair_id;
document.getElementById('progress').textContent=(current+1)+' / '+total+'；本批已完成 '+v.completed+' 对';
document.getElementById('reason').value=v.reason||'';document.querySelectorAll('input[name=label]').forEach(e=>e.checked=e.value===v.label);
document.getElementById('choices').hidden=stage==='synthetic';document.getElementById('synchoices').hidden=stage!=='synthetic';message('');
}catch(e){message(e.message)}}
function move(d){if(total)loadTask(Math.max(0,Math.min(total-1,current+d)))}
async function firstBlank(){try{let v=await request('/api/first-blank?stage='+document.getElementById('stage').value);loadTask(v.index)}catch(e){message(e.message)}}
async function save(){try{let e=document.querySelector('input[name=label]:checked');if(!task||!e)throw Error('请选择标签。');await request('/api/save',{stage:document.getElementById('stage').value,pair_id:task.pair_id,label:e.value,reason:document.getElementById('reason').value});move(1)}catch(e){message(e.message)}}
async function freezeGuide(){try{await request('/api/freeze',{});message('规范已固定，可以继续训练、验证和测试标注。')}catch(e){message(e.message)}}
loadTask(0)
</script></html>"""

class AnnotationStore:
    def __init__(self, root=ROOT):
        self.root = root
        self.meta = validate_task_manifest(root)
        self.pairs = pd.read_csv(root / "data/annotation/pair_metadata.csv", keep_default_na=False)
        self.texts = pd.read_csv(root / "data/annotation/annotation_texts.csv", keep_default_na=False).set_index("pair_id")
        self.lock = threading.RLock()
    def check_stage(self, stage):
        if stage not in ["pilot", "train", "validation", "test", "synthetic"]:
            raise GateError("Unknown annotation stage.")
        if stage != "pilot":
            require_frozen_guidelines(self.root)
        if stage == "test":
            if not (self.root / "results/tables/final_evaluation.json").exists():
                raise GateError("测试由独立盲审任务管理；建模界面仅在最终评价封存后开放只读查看。")
            from .evaluation import validate_evaluation
            validate_evaluation(self.root)
    def stage_rows(self, stage):
        self.check_stage(stage)
        if stage == "synthetic":
            p = self.root / "data/annotation/synthetic_review.csv"
            if not p.exists():
                raise GateError("完成真实标注后，运行 study synthesize 生成扰动检查。")
            return pd.read_csv(p, keep_default_na=False)
        split = "train" if stage == "pilot" else stage
        labels = pd.read_csv(labels_path(self.root, split), keep_default_na=False, dtype=str).set_index("pair_id")
        ids = self.meta["pilot_pair_ids"] if stage == "pilot" else self.pairs.loc[self.pairs.split == split, "pair_id"].tolist()
        return self.texts.loc[ids].join(labels).reset_index()
    def task(self, stage, index):
        with self.lock:
            rows = self.stage_rows(stage)
            if not 0 <= index < len(rows):
                raise GateError("Invalid task index.")
            row = rows.iloc[index]
            return {"index": index, "total": len(rows),
                    "completed": int((rows.label.ne("") & rows.reason.str.strip().ne("")).sum()),
                    "pair_id": row.pair_id, "text_a": row.text_a, "text_b": row.text_b,
                    "label": row.label, "reason": row.reason}
    def save(self, stage, pairid, label, reason):
        with self.lock:
            if stage == "test":
                raise GateError("测试标签由独立流程封存，当前界面不可修改。")
            rows = self.stage_rows(stage)
            if pairid not in set(rows.pair_id):
                raise GateError("Task ID is outside selected stage.")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
                raise GateError("请填写不超过1000字符的判断理由。")
            if stage == "synthetic":
                if label not in ["approve", "reject"]:
                    raise GateError("Invalid synthetic judgment.")
                if (self.root / "models/model_lock.json").exists():
                    raise GateError("模型已锁定；扰动判断需要显式解锁后修改。")
                p = self.root / "data/annotation/synthetic_review.csv"
                allrows = pd.read_csv(p, keep_default_na=False)
            else:
                if label not in LABELS:
                    raise GateError("Invalid label.")
                row = rows[rows.pair_id == pairid].iloc[0]
                if label == "exact" and normalize_text(row.text_a) != normalize_text(row.text_b):
                    raise GateError("两段规范化正文不同，请重新检查完全重复标签。")
                split = "train" if stage == "pilot" else stage
                if split in ["train", "validation"] and (self.root / "models/model_lock.json").exists():
                    raise GateError("模型和阈值已锁定；训练/验证标签不可继续修改。")
                if split == "test" and (self.root / "results/tables/final_evaluation.json").exists():
                    raise GateError("最终测试已封存；修改需建立新的评价版本。")
                p = labels_path(self.root, split)
                allrows = pd.read_csv(p, keep_default_na=False, dtype=str)
            mask = allrows.pair_id == pairid
            allrows.loc[mask, "label"] = label
            allrows.loc[mask, "reason"] = reason.strip()
            allrows.loc[mask, "annotation_version"] = self.meta["dataset_version"]
            write_csv(p, allrows)
            return {"saved": True}
    def freeze(self):
        with self.lock:
            if (self.root / "models/model_lock.json").exists():
                raise GateError("模型已锁定，不能重新冻结或改写标注规范。")
            pilot = self.stage_rows("pilot")
            if pilot.label.eq("").any() or pilot.reason.str.strip().eq("").any():
                raise GateError("第一批60对尚未完成，不能固定规范。")
            guide = self.root / "report/annotation_guide.md"
            write_json(self.root / "data/annotation/guideline_freeze.json",
                {"dataset_version": self.meta["dataset_version"], "guide_sha256": sha_file(guide),
                 "pilot_pair_ids": self.meta["pilot_pair_ids"], "pair_table_sha256": self.meta["pair_table_sha256"]})
            return {"frozen": True}

def serve(root: Path=ROOT, port=8765):
    store = AnnotationStore(root)
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def json_response(self, value, code=200):
            blob=json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code);self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(blob)));self.end_headers();self.wfile.write(blob)
        def do_GET(self):
            try:
                parsed=urlparse(self.path)
                if parsed.path == "/":
                    blob=PAGE.replace("__TOKEN__",token).encode("utf-8")
                    self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8")
                    self.send_header("Content-Length",str(len(blob)));self.end_headers();self.wfile.write(blob)
                    return
                args=parse_qs(parsed.query)
                stage=args.get("stage",["pilot"])[0]
                if parsed.path == "/api/task":
                    self.json_response(store.task(stage,int(args.get("index",["0"])[0])))
                elif parsed.path == "/api/first-blank":
                    rows=store.stage_rows(stage)
                    blank=rows.label.eq("") | rows.reason.str.strip().eq("")
                    index=int(next(iter(rows.index[blank]),0))
                    self.json_response({"index":index})
                elif parsed.path == "/api/status":
                    self.json_response(status(root))
                else:self.json_response({"error":"Not found"},404)
            except (ValueError, KeyError, FileNotFoundError) as e:self.json_response({"error":str(e)},400)
        def do_POST(self):
            try:
                if self.headers.get("X-CSRF-Token") != token:
                    self.json_response({"error":"Invalid session token"},403);return
                length=int(self.headers.get("Content-Length","0"))
                if not 0 <= length <= 10000:raise ValueError("Request too large")
                body=json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/save":
                    self.json_response(store.save(body["stage"],body["pair_id"],body["label"],body["reason"]))
                elif self.path == "/api/freeze":self.json_response(store.freeze())
                else:self.json_response({"error":"Not found"},404)
            except (ValueError,KeyError,FileNotFoundError) as e:self.json_response({"error":str(e)},400)
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    print(f"Annotation interface: http://localhost:{port}  (loopback only)",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
