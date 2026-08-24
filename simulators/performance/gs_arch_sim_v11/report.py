from __future__ import annotations

import csv
import html
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _scene_label(scene: str) -> str:
    labels = {
        "train": "Train",
        "lego": "Lego",
        "hotdog": "Hotdog",
        "truck": "Truck",
        "drjohnson": "DrJohnson",
        "playroom": "Playroom",
        "bicycle": "Mip-NeRF 360 Bicycle",
        "garden": "Mip-NeRF 360 Garden",
    }
    return labels.get(scene, scene)


def build_report(result_dir: Path, out: Path) -> None:
    scenes = _rows(result_dir / "scene_summary.csv")
    overall = _rows(result_dir / "overall_summary.csv")
    comparison = _rows(result_dir / "comparison_2x2.csv")
    max_fps = max(float(row["with_t_fps"]) for row in scenes)
    cards = "".join(
        f'<div class="card"><h3>{html.escape(row["allocator"])}</h3>'
        f'<p><b>{float(row["equal_scene_with_t_fps"]):.2f}</b> FPS</p>'
        f'<p>T speedup {float(row["equal_scene_t_speedup"]):.3f}× · '
        f'{int(row["camera_count"])} cameras</p></div>' for row in overall
    )
    combined = next(row for row in comparison if row["allocator"] == "atae_snugbox" and row["tile_tstop"] == "on")
    cards += (f'<div class="card"><h3>ATAE + 标准 T</h3>'
              f'<p><b>{float(combined["speedup_vs_baseline_aabb_no_t"]):.2f}×</b></p>'
              f'<p>相对 Baseline-AABB、无 T 的组合加速</p></div>')
    chart = "".join(
        f'<div class="barrow"><span>{html.escape(_scene_label(row["scene"]))}<small>{html.escape(row["allocator"])}</small></span>'
        f'<i style="width:{100*float(row["with_t_fps"])/max_fps:.2f}%"></i>'
        f'<b>{float(row["with_t_fps"]):.1f}</b></div>' for row in scenes
    )
    table = "".join(
        f'<tr><td>{html.escape(_scene_label(row["scene"]))}</td><td>{html.escape(row["allocator"])}</td>'
        f'<td>{int(row["camera_count"])}</td><td>{float(row["no_t_fps"]):.2f}</td>'
        f'<td>{float(row["with_t_fps"]):.2f}</td><td>{float(row["t_speedup"]):.3f}×</td>'
        f'<td>{100*float(row["t_stopped_fraction"]):.2f}%</td></tr>' for row in scenes
    )
    matrix = "".join(
        f'<tr><td>{html.escape(row["allocator"])}</td><td>{"开启" if row["tile_tstop"] == "on" else "关闭"}</td>'
        f'<td>{float(row["raster_fps"]):.2f}</td><td>{float(row["speedup_vs_baseline_aabb_no_t"]):.3f}×</td></tr>'
        for row in comparison
    )
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SARD v1.1.1 真实 8 场景结果与交接</title><style>
    :root{{--ink:#172033;--blue:#246bfd;--soft:#eef4ff;--line:#dbe3ef;--green:#0b9b6f;--orange:#f59e0b}}*{{box-sizing:border-box}}body{{margin:0;background:#f7f9fc;color:var(--ink);font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:auto;padding:38px 24px}}h1{{font-size:clamp(30px,5vw,56px);line-height:1.12;margin:.2em 0}}h2{{margin-top:48px}}h3{{margin:.2em 0}}.lead{{font-size:20px;max-width:850px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}}.card,.panel{{background:white;border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 8px 28px #24406b0d}}.card b{{font-size:42px;color:var(--blue)}}.flow{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}.flow div{{background:var(--soft);border-radius:14px;padding:14px 18px}}.flow em{{font-size:26px;color:var(--blue)}}.barrow{{display:grid;grid-template-columns:190px 1fr 70px;align-items:center;gap:10px;margin:10px 0}}.barrow span small{{display:block;color:#68758b}}.barrow i{{display:block;height:18px;border-radius:9px;background:linear-gradient(90deg,var(--blue),#59b4ff)}}table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}.scroll{{overflow:auto}}code{{background:#edf1f7;padding:.15em .35em;border-radius:5px}}pre{{white-space:pre-wrap;background:#172033;color:#e8eefb;border-radius:14px;padding:16px;overflow:auto}}.note{{border-left:5px solid var(--green)}}.warn{{border-left:5px solid var(--orange)}}svg{{width:100%;height:auto}}a{{color:var(--blue)}}@media(max-width:650px){{.barrow{{grid-template-columns:115px 1fr 55px}}main{{padding:24px 14px}}}}</style><main>
    <p>SARD · gs_arch_sim v1.1.1</p><h1>连续 W16 + 整 Tile T 早停<br>真实 Camera-level Raster 结果</h1>
    <p class="lead">这份报告回答一件明确的事：真实 renderer 给出的深度有序 KVP，在四个连续 W16 SARD core 上需要多少 raster 周期。覆盖 8 个场景、581 个真实 camera、双 allocator，共 1162 次 camera-allocator 重放。</p>
    <div class="grid">{cards}</div>
    <h2>电路数据流：一张图看懂</h2><div class="panel"><svg viewBox="0 0 1040 270" role="img" aria-label="SARD v1.1 dataflow"><defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#246bfd"/></marker></defs><g font-family="Microsoft YaHei,system-ui" text-anchor="middle"><rect x="20" y="75" width="150" height="110" rx="18" fill="#eef4ff" stroke="#246bfd"/><text x="95" y="110" font-size="18" font-weight="700">真实 Camera</text><text x="95" y="140" font-size="14">AABB / ATAE</text><text x="95" y="162" font-size="14">深度有序 KVP</text><rect x="220" y="75" width="170" height="110" rx="18" fill="#eafaf5" stroke="#0b9b6f"/><text x="305" y="108" font-size="18" font-weight="700">RIE</text><text x="305" y="138" font-size="14">16 行递推并行</text><text x="305" y="160" font-size="14">每周期 16 个 σ</text><rect x="440" y="75" width="150" height="110" rx="18" fill="#fff8e8" stroke="#f59e0b"/><text x="515" y="108" font-size="18" font-weight="700">Context ×2</text><text x="515" y="138" font-size="14">FIFO-32</text><text x="515" y="160" font-size="14">吸收短期错位</text><rect x="640" y="75" width="165" height="110" rx="18" fill="#f2edff" stroke="#7c3aed"/><text x="722" y="108" font-size="18" font-weight="700">W16 RSPA</text><text x="722" y="138" font-size="14">16 lane / cycle</text><text x="722" y="160" font-size="14">每 KVP 16 cycles</text><rect x="855" y="75" width="165" height="110" rx="18" fill="#eef4ff" stroke="#246bfd"/><text x="937" y="108" font-size="18" font-weight="700">4 Core 调度</text><text x="937" y="138" font-size="14">最早空闲优先</text><text x="937" y="160" font-size="14">输出 Camera cycles</text><g stroke="#246bfd" stroke-width="4" marker-end="url(#a)"><path d="M170 130 H210"/><path d="M390 130 H430"/><path d="M590 130 H630"/><path d="M805 130 H845"/></g><path d="M722 195 C722 245 305 245 305 195" fill="none" stroke="#0b9b6f" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#a)"/><text x="515" y="250" font-size="14" fill="#0b9b6f">KVP 边界写回后，下一 KVP 才能更新同一 tile 状态</text></g></svg></div>
    <div class="panel flow"><div>真实 3DGS camera</div><em>→</em><div>每 tile 的深度有序 KVP</div><em>→</em><div>全部有效像素 T&lt;10<sup>−4</sup> 后停止后续 KVP</div><em>→</em><div>连续 W16</div><em>→</em><div>4 core 最早空闲调度</div></div>
    <div class="panel note"><b>证据边界：</b>T 是标准 renderer 行为，不是 SARD 提出的贡献。仿真器采用目标连续流架构；FPS 是 400 MHz 下的 camera-level raster 分析结果，不是当前 RTL 的端到端实测 FPS。</div>
    <h2>收益到底来自哪里？</h2><div class="grid"><div class="card"><h3>ATAE / SNUGBOX</h3><p><b>2.74×</b></p><p>73.60 → 201.40 FPS。主要收益来自少生成无效 tile-KVP。</p></div><div class="card"><h3>标准 Tile-T</h3><p><b>1.16×</b></p><p>ATAE 下 201.40 → 232.82 FPS。它是 renderer 标准行为，不是 SARD 新机制。</p></div><div class="card"><h3>组合</h3><p><b>3.16×</b></p><p>AABB 无 T → ATAE + T。不能把整个 3.16× 都归因给 SARD 电路。</p></div></div>
    <h2>Allocator × Tile-T：2×2</h2><div class="scroll"><table><thead><tr><th>Allocator</th><th>Tile-T</th><th>Raster FPS</th><th>相对基线</th></tr></thead><tbody>{matrix}</tbody></table></div>
    <h2>各场景 T 后 Raster FPS</h2><div class="panel">{chart}</div>
    <h2>完整结果</h2><div class="scroll"><table><thead><tr><th>场景</th><th>Allocator</th><th>Camera</th><th>无 T FPS</th><th>T 后 FPS</th><th>T 加速</th><th>停止 KVP</th></tr></thead><tbody>{table}</tbody></table></div>
    <h2>模型公式</h2><div class="panel"><code>C_tile=max(C_RIE,16×N_KVP)+30</code>；tile 按 renderer 顺序分配给最早空闲的四个 core，camera cycles 是最慢 core 的完成时间。正式配置固定为 16×16 tile、4 core、400 MHz、2 KVP context、FIFO-32。</div>
    <div class="panel warn"><b>为什么 RIE 基本被盖住：</b>每个 KVP 的真实 producer count 在 1–16 cycles，而连续 W16 RSPA 固定需要 16 cycles/KVP。因此稳态最大值通常由 RSPA 决定；FIFO 与双 context 的作用是让两边重叠，不是凭空减少 16×N 的 blending 工作。</div>
    <h2>接手与复现</h2><div class="panel"><p>公开包只需要 Python 标准库。真实模型与数据不随包分发，使用者通过 adapter 在合法获得的数据集上重新生成汇总。</p><pre>python -m pip install -e .
sard-v11 validate-camera workloads/baseline_aabb/lego/tiles.csv
sard-v11 replay-camera workloads/baseline_aabb/lego/tiles.csv --camera 0
sard-v11 run-experiments workloads --out results
sard-v11 report results --out results/report.html</pre><p><a href="camera_results.csv">逐 camera 结果</a> · <a href="scene_summary.csv">逐场景汇总</a> · <a href="comparison_2x2.csv">2×2 表</a> · <a href="manifest.json">可追溯 manifest</a></p></div>
    <h2>最终边界</h2><div class="panel note"><ul><li>正式结果是 target streaming architecture，不是当前串行控制 RTL 的实测。</li><li>FPS 是 camera-level SARD raster FPS；预处理、排序、缓存、DRAM、主机时间和能耗均未混入。</li><li>T 终止只删除尚未开始的后续 KVP；导致最后一个像素完成的当前 KVP仍然保留。</li><li>v1.1 只保存 tile 级汇总，不保存细粒度事件轨迹，也不做可变宽度探索。</li></ul></div>
    </main></html>'''
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
